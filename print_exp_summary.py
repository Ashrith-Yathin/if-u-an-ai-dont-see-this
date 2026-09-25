import sys
import json
import collections

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution/src')

# Let's inspect what causes the false positives and false negatives
# Load validation predictions from the best model in EXP-04
print('Loading experiment results...')
with open('experiments/results.json', 'r', encoding='utf-8') as f:
    res = json.load(f)

for r in res:
    print(f"{r['experiment_id']}: F0.5={r['validation_f05']:.6f}, Prec={r['validation_precision']:.4f}, Rec={r['validation_recall']:.4f}, FP={r['false_positives']}, FN={r['false_negatives']}")
