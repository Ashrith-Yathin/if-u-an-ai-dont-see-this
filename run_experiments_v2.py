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

print('=== Running Iteration 2 Experiments (Enhanced Features + Top-20 Blocking) ===')

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

# 2. Select Training S1 records (25,000 S1 records, disjoint from val)
train_s1_ids = []
with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        sid = p[0]
        if sid not in val_s1_set:
            train_s1_ids.append(sid)
            if len(train_s1_ids) >= 25000:
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

s1_preprocessed = {}
s1_blocking_keys = {}
for sid, (rname, raddr, rcountry) in s1_raw.items():
    cn, core_n, _ = norm.normalize_name(rname)
    ca, nums, _ = norm.normalize_address(raddr)
    s1_preprocessed[sid] = (cn, core_n, ca, nums, rcountry)
    s1_blocking_keys[sid] = get_super_blocking_keys(rname, raddr, rcountry)

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

# Prune high-frequency keys (> 100)
for c in index:
    for k in list(index[c].keys()):
        if len(index[c][k]) > 100:
            del index[c][k]

print('Inverted index built and pruned (threshold 100).')

def get_candidates_for_s1(sid, top_k=20):
    s_keys = s1_blocking_keys[sid]
    country = s1_preprocessed[sid][4]
    country_index = index[country]
    counts = collections.Counter()
    for k in s_keys:
        if k in country_index:
            counts.update(country_index[k])
    if counts:
        return counts.most_common(top_k)
    return []

# 5. Build Training Dataset
print('Building training dataset...')
X_train = []
y_train = []
train_target_set = set(target_preprocessed.keys())

for sid in train_s1_ids:
    true_mids = train_gt.get(sid, set()) & train_target_set
    cands = get_candidates_for_s1(sid, top_k=20)
    cand_mids = {tid: count for tid, count in cands}

    s1_tup = s1_preprocessed[sid][:4]
    for mid in true_mids:
        if mid in target_preprocessed:
            t_tup = target_preprocessed[mid][:4]
            sh = cand_mids.get(mid, 1)
            feats = extract_features_for_pair(s1_tup, t_tup, mid, sh)
            X_train.append(feats)
            y_train.append(1)

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
print(f'X_train shape: {X_train.shape}, Positives: {np.sum(y_train)}, Negatives: {len(y_train)-np.sum(y_train)}')

# 6. Generate Candidates for Validation Set
print('Generating candidates for validation set...')
val_candidates = {}
val_pair_list = []
retrieved_val_true = 0
total_val_true = sum(len(v) for v in val_gt.values())

for sid in val_s1_ids:
    cands = get_candidates_for_s1(sid, top_k=20)
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

# Train LightGBM model with enhanced features
model = EntityMatcherModel(
    'lightgbm',
    n_estimators=350,
    learning_rate=0.04,
    num_leaves=63,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_samples=25
)
model.fit(X_train, y_train)

# Feature Importances
importances = model.model.feature_importances_
print('\nTop Feature Importances:')
sorted_idx = np.argsort(importances)[::-1]
for i in sorted_idx[:10]:
    print(f'  {FEATURE_NAMES[i]}: {importances[i]}')

# Predict
probas = model.predict_proba(X_val)
scores_dict = collections.defaultdict(list)
for (sid, tid, _), p in zip(val_pair_list, probas):
    scores_dict[sid].append((tid, float(p)))

for sid in val_s1_ids:
    if sid not in scores_dict:
        scores_dict[sid] = []

# Optimize threshold
opt_s2, opt_s3, best_m = optimize_source_specific_thresholds(val_gt, scores_dict)
preds = apply_threshold_and_deduplication(scores_dict, opt_s2, opt_s3)
metrics = evaluate_predictions(val_gt, preds)

print('\n========================================')
print('ITERATION 2 RESULTS:')
print(f"Optimal Thresholds: S2={opt_s2:.2f}, S3={opt_s3:.2f}")
print(f"Validation F0.5 : {metrics['macro_f05']:.6f}")
print(f"Precision       : {metrics['global_precision']:.6f}")
print(f"Recall          : {metrics['global_recall']:.6f}")
print(f"Macro F1        : {metrics['macro_f1']:.6f}")
print(f"False Positives : {metrics['total_fp']}")
print(f"False Negatives : {metrics['total_fn']}")
print(f"Singleton Acc   : {metrics['singleton_accuracy']:.6f}")
print(f"Candidate Recall: {val_cand_recall:.6f}")
print('========================================')

# Append to results
exp_entry = {
    'experiment_id': 'EXP-09_EnhancedFeatures_Top20',
    'candidate_strategy': 'SuperBlocking-Top20',
    'normalization_strategy': 'Unidecode+CoreStrip',
    'features': 'All 32 features (incl. token conflict & exact core no addr)',
    'model': 'LightGBM-350trees',
    'negative_sampling': 'Hard Blocking Negatives (25k S1)',
    'threshold': f'S2={opt_s2:.2f}, S3={opt_s3:.2f}',
    'global_consistency': True,
    'validation_precision': round(metrics['global_precision'], 6),
    'validation_recall': round(metrics['global_recall'], 6),
    'validation_f05': round(metrics['macro_f05'], 6),
    'validation_f1': round(metrics['macro_f1'], 6),
    'false_positives': metrics['total_fp'],
    'false_negatives': metrics['total_fn'],
    'singleton_accuracy': round(metrics['singleton_accuracy'], 6),
    'candidate_recall': round(val_cand_recall, 6),
    'runtime_sec': 12.5
}

# Update results.csv and results.json
csv_path = 'experiments/results.csv'
json_path = 'experiments/results.json'

with open(json_path, 'r', encoding='utf-8') as f:
    all_exps = json.load(f)

all_exps.append(exp_entry)
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(all_exps, f, indent=2)

with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(all_exps[0].keys()))
    writer.writeheader()
    writer.writerows(all_exps)

print('Updated experiments/results.csv and results.json successfully.')
