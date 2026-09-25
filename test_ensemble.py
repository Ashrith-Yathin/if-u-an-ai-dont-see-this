import sys
import collections
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution/src')

from run_experiments_v2 import (
    X_train, y_train, X_val, val_pair_list, val_gt, val_s1_ids,
    EntityMatcherModel, optimize_source_specific_thresholds,
    apply_threshold_and_deduplication, evaluate_predictions
)

print('Training LightGBM...')
m_lgb = EntityMatcherModel(
    'lightgbm',
    n_estimators=350,
    learning_rate=0.04,
    num_leaves=63,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85
)
m_lgb.fit(X_train, y_train)
p_lgb = m_lgb.predict_proba(X_val)

print('Training HistGradientBoosting...')
m_hgb = EntityMatcherModel('hist_gb')
m_hgb.fit(X_train, y_train)
p_hgb = m_hgb.predict_proba(X_val)

# Ensemble blend: 0.6 * LightGBM + 0.4 * HistGradientBoosting
p_ens = 0.6 * p_lgb + 0.4 * p_hgb

scores_dict = collections.defaultdict(list)
for (sid, tid, _), p in zip(val_pair_list, p_ens):
    scores_dict[sid].append((tid, float(p)))

for sid in val_s1_ids:
    if sid not in scores_dict:
        scores_dict[sid] = []

opt_s2, opt_s3, _ = optimize_source_specific_thresholds(val_gt, scores_dict)
preds = apply_threshold_and_deduplication(scores_dict, opt_s2, opt_s3)
metrics = evaluate_predictions(val_gt, preds)

print('\n========================================')
print('ENSEMBLE (LightGBM + HistGB) RESULTS:')
print(f"Optimal Thresholds: S2={opt_s2:.2f}, S3={opt_s3:.2f}")
print(f"Validation F0.5 : {metrics['macro_f05']:.6f}")
print(f"Precision       : {metrics['global_precision']:.6f}")
print(f"Recall          : {metrics['global_recall']:.6f}")
print(f"Macro F1        : {metrics['macro_f1']:.6f}")
print(f"False Positives : {metrics['total_fp']}")
print(f"False Negatives : {metrics['total_fn']}")
print(f"Singleton Acc   : {metrics['singleton_accuracy']:.6f}")
print('========================================')
