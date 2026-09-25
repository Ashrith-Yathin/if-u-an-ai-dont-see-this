import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution/src')

import normalization as norm
from features import extract_features_for_pair, FEATURE_NAMES
from model import EntityMatcherModel
from run_experiments import (
    val_s1_ids, val_gt, s1_preprocessed, target_preprocessed,
    val_pair_list, X_train, y_train, X_val, opt_s2, opt_s3
)

# Train model
m = EntityMatcherModel('lightgbm', n_estimators=300, learning_rate=0.04, num_leaves=45)
m.fit(X_train, y_train)

probas = m.predict_proba(X_val)

# Inspect false negatives: true matches that got low probability
fn_examples = []
fp_examples = []

for (sid, tid, feats), p in zip(val_pair_list, probas):
    is_true = tid in val_gt.get(sid, set())
    t = opt_s2 if tid.startswith('S2-') else opt_s3
    if is_true and p < t:
        s1_cn, s1_core, s1_ca, s1_nums, s1_c = s1_preprocessed[sid]
        t_cn, t_core, t_ca, t_nums, t_c = target_preprocessed[tid]
        fn_examples.append((sid, tid, p, s1_cn, t_cn, s1_ca, t_ca, feats))
    elif not is_true and p >= t:
        s1_cn, s1_core, s1_ca, s1_nums, s1_c = s1_preprocessed[sid]
        t_cn, t_core, t_ca, t_nums, t_c = target_preprocessed[tid]
        fp_examples.append((sid, tid, p, s1_cn, t_cn, s1_ca, t_ca, feats))

print(f'Total FN in candidate pairs: {len(fn_examples)}')
print(f'Total FP in candidate pairs: {len(fp_examples)}')

print('\nTop 5 False Negatives (scored just below threshold):')
fn_examples.sort(key=lambda x: x[2], reverse=True)
for ex in fn_examples[:5]:
    print(f'S1: {ex[0]} -> {ex[1]} | Prob: {ex[2]:.4f}')
    print(f'  NAME S1: {ex[3]} | T: {ex[4]}')
    print(f'  ADDR S1: {ex[5]} | T: {ex[6]}')
    print()

print('\nTop 5 False Positives (wrong pairs scored high):')
fp_examples.sort(key=lambda x: x[2], reverse=True)
for ex in fp_examples[:5]:
    print(f'S1: {ex[0]} -> {ex[1]} | Prob: {ex[2]:.4f}')
    print(f'  NAME S1: {ex[3]} | T: {ex[4]}')
    print(f'  ADDR S1: {ex[5]} | T: {ex[6]}')
    print()
