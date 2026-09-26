"""
End-to-end test-set inference: load -> block -> feature -> score -> threshold
-> write both required TSVs.

`candidate_pairs.tsv` here is written as the *actual* input to the matcher
(exactly the candidates the model scored), matching the PS's requirement
that it be the last-stage set, not an earlier looser blocking pass.

Feature notes (auto-inherited from features.py — no changes needed here):
  - number_overlap and addr_a/b_missing are computed inside pair_features(),
    so they are produced identically at train and inference time as long as
    both use the same features.py.  No extra wiring required.

Usage:
    python inference_pipeline.py --model outputs_model.txt --threshold_file outputs_threshold.txt
"""

import argparse
import os
import numpy as np
import lightgbm as lgb
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from config import CFG
from data_utils import load_source, write_matching_results, write_candidate_pairs
from blocking import generate_candidates
from features import fit_tfidf_on_all_text, build_feature_matrix, df_to_lookup
import torch
from train_matcher import MLP, get_mlp_preds


def _fit_tfidf_memory_efficient(*paths) -> TfidfVectorizer:
    """
    Fit TF-IDF without holding all DataFrames in memory simultaneously.
    Reads each source, extracts text, then releases the DataFrame before
    moving to the next one.  On a 32 GB machine this avoids the ~18 GB
    peak caused by keeping all 6 source DFs alive at once.
    """
    print("  Fitting TF-IDF (memory-efficient streaming across all sources)...")
    all_text = []
    for path in paths:
        print(f"    collecting text from {os.path.basename(path)}...")
        df = load_source(path)
        if len(df) > CFG.TFIDF_SAMPLE_SIZE:
            df = df.sample(n=CFG.TFIDF_SAMPLE_SIZE, random_state=42)
        all_text.extend((df["norm_name"] + " " + df["norm_addr"]).tolist())
        del df   # release immediately
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20000)
    vec.fit(all_text)
    del all_text
    print("  TF-IDF fitted.")
    return vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                     help="path to saved model (outputs_model.json for XGBoost, outputs_model.txt for LightGBM)")
    ap.add_argument("--threshold_file", default="outputs_threshold.txt")
    ap.add_argument("--threshold", type=float, default=None,
                     help="override the tuned threshold if you want to trade precision/recall manually")
    args = ap.parse_args()

    threshold = args.threshold
    if threshold is None:
        with open(args.threshold_file) as f:
            threshold = float(f.read().strip())
    print(f"Using decision threshold: {threshold}")

    # Fit TF-IDF on train+test text so vocabulary covers France (test-only
    # country).  DFs are released after text extraction to minimise peak RAM.
    print("Fitting TF-IDF across all sources (train + test)...")
    tfidf_vec = _fit_tfidf_memory_efficient(
        CFG.TRAIN_S1, CFG.TRAIN_S2, CFG.TRAIN_S3,
        CFG.TEST_S1,  CFG.TEST_S2,  CFG.TEST_S3,
    )

    print("Loading test sources...")
    s1 = load_source(CFG.TEST_S1)
    if CFG.TEST_ENTITY_SAMPLE and CFG.TEST_ENTITY_SAMPLE < len(s1):
        print(f"Subsampling test set to {CFG.TEST_ENTITY_SAMPLE} entities for fast dry-run...")
        s1 = s1.sample(n=CFG.TEST_ENTITY_SAMPLE, random_state=42).reset_index(drop=True)
        
    s2 = load_source(CFG.TEST_S2)
    s3 = load_source(CFG.TEST_S3)

    print("Generating candidates for the full test Source-1 set...")
    cache_test_prefix = os.path.join(CFG.CACHE_DIR, "candidates_test")
    candidates = generate_candidates(s1, s2, s3, cache_prefix=cache_test_prefix)

    # Every Source-1 entity must appear, even with an empty candidate/match list.
    all_s1_ids = set(s1["entity_id"])
    for s1_id in all_s1_ids:
        candidates.setdefault(s1_id, set())

    write_candidate_pairs(candidates)
    print(f"Wrote {CFG.CANDIDATES_OUT} ({sum(len(v) for v in candidates.values())} candidate pairs)")

    s1_lookup = df_to_lookup(s1)
    # Only index S2/S3 rows that are actual candidates — avoids 8 GB dict OOM
    reachable_ids = set(cid for cids in candidates.values() for cid in cids)
    s2_reach = s2[s2["entity_id"].isin(reachable_ids)]
    s3_reach = s3[s3["entity_id"].isin(reachable_ids)]
    other_lookup = {**df_to_lookup(s2_reach), **df_to_lookup(s3_reach)}
    del s2_reach, s3_reach, s2, s3
    import gc; gc.collect()

    print("Flattening candidate pairs for scoring...")
    pairs = [(s1_id, cid) for s1_id, cids in tqdm(candidates.items(), desc="  Flattening pairs") for cid in cids]
    print(f"Scoring {len(pairs)} candidate pairs...")

    print("Loading Blended Models (XGBoost + PyTorch MLP)...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("outputs_xgb_model.json")
    
    mlp_stats = np.load("outputs_mlp_stats.npy", allow_pickle=True).item()
    mlp_mean, mlp_std = mlp_stats["mean"], mlp_stats["std"]

    matches = {s1_id: set() for s1_id in all_s1_ids}

    if pairs:
        X, valid_pairs = build_feature_matrix(pairs, s1_lookup, other_lookup, tfidf_vec)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_model = MLP(X.shape[1]).to(device)
        mlp_model.load_state_dict(torch.load("outputs_mlp_model.pth", map_location=device))
        
        print("Scoring candidate pairs with blended models...")
        xgb_preds = xgb_model.predict_proba(X)[:, 1] if len(X) else np.array([])
        mlp_preds = get_mlp_preds(mlp_model, X, mlp_mean, mlp_std) if len(X) else np.array([])
        scores = (xgb_preds + mlp_preds) / 2.0
        
        # Enforce one-owner rule: target candidate belongs to the S1 that scores it highest
        scored_pairs = sorted(zip(valid_pairs, scores), key=lambda x: x[1], reverse=True)
        assigned_others = set()
        
        for (s1_id, other_id), score in tqdm(scored_pairs, desc="  Filtering matches by threshold (One-Owner)"):
            if score >= threshold:
                if other_id not in assigned_others:
                    matches[s1_id].add(other_id)
                    assigned_others.add(other_id)

    write_matching_results(matches)
    n_matched_entities = sum(1 for v in matches.values() if v)
    print(f"Wrote {CFG.MATCHING_OUT} ({n_matched_entities}/{len(matches)} S1 entities have >=1 match)")


if __name__ == "__main__":
    main()
