import sys
import os
import time
import json
import csv
import collections
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution/src')

import normalization as norm
from features import extract_features_for_pair, FEATURE_NAMES
from model import EntityMatcherModel
from evaluation import evaluate_predictions
from thresholding import optimize_global_threshold, optimize_source_specific_thresholds, apply_threshold_and_deduplication
from test_super_blocking import get_super_blocking_keys

print('=== Starting Business Entity Resolution Experiment Suite ===')

# 1. Load Validation S1 set (5000 S1 records)
val_s1_ids = []
with open('experiments/val_s1_ids.txt', 'r', encoding='utf-8') as f:
    for line in f:
        val_s1_ids.append(line.strip())
        if len(val_s1_ids) >= 5000:
            break

val_s1_set = set(val_s1_ids)

# Load ground truth for val_s1_ids
val_gt = {}
val_true_targets = set()
with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in val_s1_set:
            mids = p[1].split(',') if len(p) > 1 and p[1] else []
            val_gt[p[0]] = set(mids)
            val_true_targets.update(mids)

print(f'Validation set: {len(val_s1_ids)} S1 records, {sum(len(v) for v in val_gt.values())} true positive links.')

# 2. Select Training S1 records (disjoint from validation set)
train_s1_ids = []
with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        sid = p[0]
        if sid not in val_s1_set:
            train_s1_ids.append(sid)
            if len(train_s1_ids) >= 15000:
                break

train_s1_set = set(train_s1_ids)
train_gt = {}
train_true_targets = set()
with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in train_s1_set:
            mids = p[1].split(',') if len(p) > 1 and p[1] else []
            train_gt[p[0]] = set(mids)
            train_true_targets.update(mids)

print(f'Training sample: {len(train_s1_ids)} S1 records, {sum(len(v) for v in train_gt.values())} true positive links.')

# 3. Load S1 data for train and val
all_needed_s1 = val_s1_set | train_s1_set
s1_raw = {}
with open('student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in all_needed_s1:
            s1_raw[p[0]] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else '')

# Preprocess S1 records
# (clean_name, core_name, clean_addr, nums_set, country)
s1_preprocessed = {}
for sid, (rname, raddr, rcountry) in s1_raw.items():
    cn, core_n, _ = norm.normalize_name(rname)
    ca, nums, _ = norm.normalize_address(raddr)
    s1_preprocessed[sid] = (cn, core_n, ca, nums, rcountry)

# 4. Load Target records (True targets + 50,000 distractors)
all_needed_targets = val_true_targets | train_true_targets
target_raw = {}
distractors_s2_count = 0
with open('student_resource/dataset/train/train_source2.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        tid = p[0]
        if tid in all_needed_targets or distractors_s2_count < 35000:
            target_raw[tid] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else '')
            if tid not in all_needed_targets:
                distractors_s2_count += 1

distractors_s3_count = 0
with open('student_resource/dataset/train/train_source3.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        tid = p[0]
        if tid in all_needed_targets or distractors_s3_count < 35000:
            target_raw[tid] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else '')
            if tid not in all_needed_targets:
                distractors_s3_count += 1

print(f'Total target pool: {len(target_raw):,} records loaded.')

target_preprocessed = {}
target_keys = {}
for tid, (rname, raddr, rcountry) in target_raw.items():
    cn, core_n, _ = norm.normalize_name(rname)
    ca, nums, _ = norm.normalize_address(raddr)
    target_preprocessed[tid] = (cn, core_n, ca, nums, rcountry)
    target_keys[tid] = get_super_blocking_keys(rname, raddr, rcountry)

# Build Inverted Index by Country
index = collections.defaultdict(lambda: collections.defaultdict(list))
for tid, tkeys in target_keys.items():
    rcountry = target_preprocessed[tid][4]
    for k in tkeys:
        index[rcountry][k].append(tid)

# Prune high-frequency keys (> 60)
for c in index:
    for k in list(index[c].keys()):
        if len(index[c][k]) > 60:
            del index[c][k]

print('Inverted index built and pruned.')

# Function to get candidates for an S1 entity
def get_candidates_for_s1(sid, top_k=15):
    cn, core_n, ca, nums, country = s1_preprocessed[sid]
    s_keys = get_super_blocking_keys(cn, ca, country)
    country_index = index[country]
    counts = collections.Counter()
    for k in s_keys:
        if k in country_index:
            counts.update(country_index[k])
    if counts:
        return counts.most_common(top_k)
    return []

# 5. Build Training Dataset (Positives + Hard Negatives)
print('Building training dataset with hard negatives...')
X_train = []
y_train = []
train_target_set = set(target_preprocessed.keys())

for sid in train_s1_ids:
    true_mids = train_gt.get(sid, set()) & train_target_set
    cands = get_candidates_for_s1(sid, top_k=15)
    cand_mids = {tid: count for tid, count in cands}

    # Add all true positives
    s1_tup = s1_preprocessed[sid][:4]
    for mid in true_mids:
        if mid in target_preprocessed:
            t_tup = target_preprocessed[mid][:4]
            sh = cand_mids.get(mid, 1)
            feats = extract_features_for_pair(s1_tup, t_tup, mid, sh)
            X_train.append(feats)
            y_train.append(1)

    # Add hard negatives from top candidates
    neg_count = 0
    for tid, sh in cands:
        if tid not in true_mids and tid in target_preprocessed:
            t_tup = target_preprocessed[tid][:4]
            feats = extract_features_for_pair(s1_tup, t_tup, tid, sh)
            X_train.append(feats)
            y_train.append(0)
            neg_count += 1
            if neg_count >= max(2, len(true_mids) * 2):
                break

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
print(f'Training dataset built: X_train shape = {X_train.shape}, positive={np.sum(y_train)}, negative={len(y_train)-np.sum(y_train)}')

# 6. Generate Candidates for Validation S1 records
print('Generating candidates for validation set...')
t_val_start = time.time()
val_candidates = {}
val_pair_list = []  # (s1_id, target_id, feats)

retrieved_val_true = 0
total_val_true = sum(len(v) for v in val_gt.values())

for sid in val_s1_ids:
    cands = get_candidates_for_s1(sid, top_k=15)
    val_candidates[sid] = [tid for tid, _ in cands]
    true_set = val_gt.get(sid, set())
    retrieved_val_true += len(true_set & set(val_candidates[sid]))
    
    s1_tup = s1_preprocessed[sid][:4]
    for tid, sh in cands:
        if tid in target_preprocessed:
            t_tup = target_preprocessed[tid][:4]
            feats = extract_features_for_pair(s1_tup, t_tup, tid, sh)
            val_pair_list.append((sid, tid, feats))

val_cand_recall = retrieved_val_true / total_val_true if total_val_true > 0 else 0
print(f'Validation candidate recall: {retrieved_val_true}/{total_val_true} = {val_cand_recall*100:.2f}%')
print(f'Total validation candidate pairs to score: {len(val_pair_list):,}')

X_val = np.array([p[2] for p in val_pair_list], dtype=np.float32)

# Experiments list
experiments = []

# Helper to run an experiment
def run_exp(exp_id, cand_strat, norm_strat, feat_desc, model_obj, neg_sampling, threshold_desc, use_dedup=False, custom_X_train=None, custom_X_val=None, s2_t=None, s3_t=None):
    t0 = time.time()
    xtr = X_train if custom_X_train is None else custom_X_train
    xval = X_val if custom_X_val is None else custom_X_val
    
    # Train
    model_obj.fit(xtr, y_train)
    # Predict
    probas = model_obj.predict_proba(xval)
    
    # Group probas by s1_id
    scores_dict = collections.defaultdict(list)
    for (sid, tid, _), p in zip(val_pair_list, probas):
        scores_dict[sid].append((tid, float(p)))
        
    for sid in val_s1_ids:
        if sid not in scores_dict:
            scores_dict[sid] = []

    # Thresholding & predictions
    if s2_t is not None and s3_t is not None:
        thresh_val = f'S2={s2_t:.2f}, S3={s3_t:.2f}'
        if use_dedup:
            preds = apply_threshold_and_deduplication(scores_dict, s2_t, s3_t)
        else:
            preds = {}
            for sid, scs in scores_dict.items():
                preds[sid] = {tid for tid, p in scs if (tid.startswith('S2-') and p >= s2_t) or (tid.startswith('S3-') and p >= s3_t)}
    elif threshold_desc == 'tune':
        best_t, _ = optimize_global_threshold(val_gt, scores_dict)
        thresh_val = f'{best_t:.2f}'
        if use_dedup:
            preds = apply_threshold_and_deduplication(scores_dict, best_t, best_t)
        else:
            preds = {sid: {tid for tid, p in scs if p >= best_t} for sid, scs in scores_dict.items()}
    else:
        fixed_t = float(threshold_desc)
        thresh_val = f'{fixed_t:.2f}'
        if use_dedup:
            preds = apply_threshold_and_deduplication(scores_dict, fixed_t, fixed_t)
        else:
            preds = {sid: {tid for tid, p in scs if p >= fixed_t} for sid, scs in scores_dict.items()}

    metrics = evaluate_predictions(val_gt, preds)
    elapsed = time.time() - t0

    exp_res = {
        'experiment_id': exp_id,
        'candidate_strategy': cand_strat,
        'normalization_strategy': norm_strat,
        'features': feat_desc,
        'model': model_obj.model_type,
        'negative_sampling': neg_sampling,
        'threshold': thresh_val,
        'global_consistency': use_dedup,
        'validation_precision': round(metrics['global_precision'], 6),
        'validation_recall': round(metrics['global_recall'], 6),
        'validation_f05': round(metrics['macro_f05'], 6),
        'validation_f1': round(metrics['macro_f1'], 6),
        'false_positives': metrics['total_fp'],
        'false_negatives': metrics['total_fn'],
        'singleton_accuracy': round(metrics['singleton_accuracy'], 6),
        'candidate_recall': round(val_cand_recall, 6),
        'runtime_sec': round(elapsed, 2)
    }
    experiments.append(exp_res)
    print(f"[{exp_id}] F0.5={exp_res['validation_f05']:.6f} | Prec={exp_res['validation_precision']:.4f} | Rec={exp_res['validation_recall']:.4f} | FP={exp_res['false_positives']} | FN={exp_res['false_negatives']} | Thresh={thresh_val} | Dedup={use_dedup} ({elapsed:.1f}s)")
    return exp_res, scores_dict

# Run Experiments Suite

# 1. Baseline: LightGBM, default threshold 0.50, no dedup
m1 = EntityMatcherModel('lightgbm', n_estimators=200, learning_rate=0.05)
run_exp('EXP-01_Baseline', 'SuperBlocking-Top15', 'Unidecode+CoreStrip', 'All 28 features', m1, 'Hard Blocking Negatives', '0.50', use_dedup=False)

# 2. Threshold Tuning
m2 = EntityMatcherModel('lightgbm', n_estimators=200, learning_rate=0.05)
_, scores_dict = run_exp('EXP-02_TunedThreshold', 'SuperBlocking-Top15', 'Unidecode+CoreStrip', 'All 28 features', m2, 'Hard Blocking Negatives', 'tune', use_dedup=False)

# 3. Global Consistency / 1-to-1 Target Deduplication
m3 = EntityMatcherModel('lightgbm', n_estimators=200, learning_rate=0.05)
run_exp('EXP-03_GlobalConsistency', 'SuperBlocking-Top15', 'Unidecode+CoreStrip', 'All 28 features', m3, 'Hard Blocking Negatives', 'tune', use_dedup=True)

# 4. Source-Specific Thresholds (S2 vs S3) + Deduplication
opt_s2, opt_s3, _ = optimize_source_specific_thresholds(val_gt, scores_dict)
m4 = EntityMatcherModel('lightgbm', n_estimators=200, learning_rate=0.05)
run_exp('EXP-04_SourceSpecificThresholds', 'SuperBlocking-Top15', 'Unidecode+CoreStrip', 'All 28 features', m4, 'Hard Blocking Negatives', 'source-specific', use_dedup=True, s2_t=opt_s2, s3_t=opt_s3)

# 5. Ablation A: Name Features Only
name_feat_indices = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith('name_')]
X_tr_name = X_train[:, name_feat_indices]
X_val_name = X_val[:, name_feat_indices]
m_name = EntityMatcherModel('lightgbm', n_estimators=200, learning_rate=0.05)
run_exp('EXP-05_Ablation_NameOnly', 'SuperBlocking-Top15', 'Unidecode+CoreStrip', 'Name Features Only (11)', m_name, 'Hard Blocking Negatives', 'tune', use_dedup=True, custom_X_train=X_tr_name, custom_X_val=X_val_name)

# 6. Ablation B: Address Features Only
addr_feat_indices = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith('addr_') or name.startswith('num_')]
X_tr_addr = X_train[:, addr_feat_indices]
X_val_addr = X_val[:, addr_feat_indices]
m_addr = EntityMatcherModel('lightgbm', n_estimators=200, learning_rate=0.05)
run_exp('EXP-06_Ablation_AddressOnly', 'SuperBlocking-Top15', 'Unidecode+CoreStrip', 'Address Features Only (10)', m_addr, 'Hard Blocking Negatives', 'tune', use_dedup=True, custom_X_train=X_tr_addr, custom_X_val=X_val_addr)

# 7. Model Comparison: HistGradientBoosting
m_hgb = EntityMatcherModel('hist_gb')
run_exp('EXP-07_HistGradientBoosting', 'SuperBlocking-Top15', 'Unidecode+CoreStrip', 'All 28 features', m_hgb, 'Hard Blocking Negatives', 'tune', use_dedup=True)

# 8. High-Capacity LightGBM (500 trees, tuned depth)
m_deep = EntityMatcherModel('lightgbm', n_estimators=400, learning_rate=0.03, num_leaves=63, max_depth=8, colsample_bytree=0.85, subsample=0.85)
best_exp, best_scores = run_exp('EXP-08_DeepLightGBM_Tuned', 'SuperBlocking-Top15', 'Unidecode+CoreStrip', 'All 28 features (Tuned)', m_deep, 'Hard Blocking Negatives', 'source-specific', use_dedup=True, s2_t=opt_s2, s3_t=opt_s3)

# Save Experiments Results to CSV and JSON
csv_path = 'experiments/results.csv'
json_path = 'experiments/results.json'

with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(experiments[0].keys()))
    writer.writeheader()
    writer.writerows(experiments)

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(experiments, f, indent=2)

print(f'\nAll experiments successfully completed and recorded to {csv_path} and {json_path}.')
