import os
import gc
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CFG
from data_utils import load_source
from features import build_feature_matrix, fit_tfidf_on_all_text, FEATURE_NAMES, df_to_lookup
from evaluate import macro_f0_5
from blocking import load_candidates_cache

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze()

def train_mlp(X_train, y_train, epochs=5, batch_size=4096):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training MLP on {device}...")
    model = MLP(X_train.shape[1]).to(device)
    
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    
    dataset = TensorDataset(torch.FloatTensor(X_train_norm), torch.FloatTensor(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-3, steps_per_epoch=len(loader), epochs=epochs
    )
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}")
        
    return model, mean, std

def get_mlp_preds(model, X_val, mean, std, batch_size=8192):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_val_norm = (X_val - mean) / std
    dataset = TensorDataset(torch.FloatTensor(X_val_norm))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    preds = []
    with torch.no_grad():
        for (X_batch,) in loader:
            p = model(X_batch.to(device))
            preds.extend(p.cpu().numpy().tolist())
    return np.array(preds)

def build_training_pairs(cand_train: dict, gt: dict, max_negatives: int, seed: int):
    rng = np.random.RandomState(seed)
    pos_pairs = []
    neg_pairs = []
    for s1_id, cids in tqdm(cand_train.items(), desc="  Building training pairs"):
        true_matches = gt.get(s1_id, set())
        matched_cids = [c for c in cids if c in true_matches]
        unmatched_cids = [c for c in cids if c not in true_matches]
        
        for c in matched_cids:
            pos_pairs.append((s1_id, c))
            
        if unmatched_cids:
            n_neg = min(len(unmatched_cids), max(1, len(matched_cids) * max_negatives))
            sampled_negs = rng.choice(unmatched_cids, n_neg, replace=False)
            for c in sampled_negs:
                neg_pairs.append((s1_id, c))
    return pos_pairs, neg_pairs

def entity_level_split(s1_ids, val_frac, seed):
    import random
    rng = random.Random(seed)
    ids = list(s1_ids)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_frac)
    return set(ids[n_val:]), set(ids[:n_val])


def tune_threshold(scores, valid_pairs, ground_truth, all_s1_ids_in_split):
    best_t, best_f0_5 = 0.5, -1.0
    for t in tqdm(np.arange(0.05, 1.0, 0.01), desc="  Sweeping F_0.5 thresholds"):
        preds = {s1_id: set() for s1_id in all_s1_ids_in_split}
        
        # Sort predictions by score descending for one-owner rule
        scored_pairs = sorted(zip(valid_pairs, scores), key=lambda x: x[1], reverse=True)
        assigned_others = set()
        
        for (s1_id, other_id), score in scored_pairs:
            if score >= t:
                if other_id not in assigned_others:
                    preds[s1_id].add(other_id)
                    assigned_others.add(other_id)
        truth_subset = {k: v for k, v in ground_truth.items() if k in all_s1_ids_in_split}
        result = macro_f0_5(preds, truth_subset)
        if result["macro_f0_5"] > best_f0_5:
            best_f0_5, best_t = result["macro_f0_5"], t
    return best_t, best_f0_5


def main():
    print("Loading datasets...")
    s1 = load_source(CFG.TRAIN_S1)
    s2 = load_source(CFG.TRAIN_S2)
    s3 = load_source(CFG.TRAIN_S3)

    print("Loading candidate dictionary...")
    cand_all = load_candidates_cache(os.path.join(CFG.CACHE_DIR, "candidates_train_combined.parquet"))
    cand_all = {str(k): set(str(c) for c in v if str(c).strip()) for k, v in cand_all.items()}

    gt_df = pd.read_csv(CFG.TRAIN_GT, sep="\t")
    gt = {str(row.source1_entity_id): set(str(row.matched_entity_ids).split(",")) 
          for row in gt_df.itertuples(index=False) if pd.notna(row.matched_entity_ids)}

    train_ids, val_ids = entity_level_split(s1["entity_id"], CFG.VAL_FRAC, CFG.SEED)
    s1_train = s1[s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    train_ids = {str(x) for x in train_ids}
    val_ids = {str(x) for x in val_ids}
    cand_train = {k: cand_all[k] for k in train_ids if k in cand_all}
    cand_val = {k: cand_all[k] for k in val_ids if k in cand_all}

    print("Fitting shared TF-IDF vectorizer...")
    tfidf_vec = fit_tfidf_on_all_text(s1, s2, s3)

    s1_lookup = df_to_lookup(s1)
    reachable_other_ids = set(cid for cids in cand_all.values() for cid in cids)
    s2_reachable = s2[s2["entity_id"].isin(reachable_other_ids)]
    s3_reachable = s3[s3["entity_id"].isin(reachable_other_ids)]
    other_lookup = {**df_to_lookup(s2_reachable), **df_to_lookup(s3_reachable)}
    del s2_reachable, s3_reachable
    import gc; gc.collect()

    print("Building training pairs...")
    pos_pairs, neg_pairs = build_training_pairs(cand_train, gt, CFG.MAX_NEGATIVES_PER_POSITIVE, CFG.SEED)
    
    X_pos, valid_pos = build_feature_matrix(pos_pairs, s1_lookup, other_lookup, tfidf_vec)
    X_neg, valid_neg = build_feature_matrix(neg_pairs, s1_lookup, other_lookup, tfidf_vec)

    X = np.vstack([X_pos, X_neg]) if len(valid_neg) else X_pos
    y = np.array([1] * len(valid_pos) + [0] * len(valid_neg))

    print("--- 1. Training MLP ---")
    mlp_model, mlp_mean, mlp_std = train_mlp(X, y)
    
    print("--- 2. Training XGBoost ---")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", 
        device="cuda" if torch.cuda.is_available() else "cpu",
        scale_pos_weight=(len(valid_neg) / max(len(valid_pos), 1)),
        eval_metric="logloss", random_state=CFG.SEED, verbosity=1
    )
    xgb_model.fit(X, y)

    print("Scoring validation candidates (13M rows vectorized!)...")
    val_pairs = [(s1_id, cid) for s1_id, cids in cand_val.items() for cid in cids]
    X_val, valid_val_pairs = build_feature_matrix(val_pairs, s1_lookup, other_lookup, tfidf_vec)
    
    print("Extracting predictions...")
    xgb_preds = xgb_model.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])
    mlp_preds = get_mlp_preds(mlp_model, X_val, mlp_mean, mlp_std) if len(X_val) else np.array([])
    
    # Simple blend average
    blend_preds = (xgb_preds + mlp_preds) / 2.0

    print("Tuning decision threshold for F_0.5 on Blended Scores...")
    best_t, best_f0_5 = tune_threshold(blend_preds, valid_val_pairs, gt, set(s1_val["entity_id"]))
    print(f"  best blend threshold={best_t:.2f}  val macro F_0.5={best_f0_5:.4f}")

    print("Saving Models...")
    xgb_model.save_model("outputs_xgb_model.json")
    torch.save(mlp_model.state_dict(), "outputs_mlp_model.pth")
    np.save("outputs_mlp_stats.npy", {"mean": mlp_mean, "std": mlp_std})
    
    with open("outputs_threshold.txt", "w") as f:
        f.write(str(best_t))
    print("Rescue completely successful!")

if __name__ == "__main__":
    main()
