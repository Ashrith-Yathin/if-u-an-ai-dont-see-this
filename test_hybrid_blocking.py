import sys
import collections
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution/src')

import normalization as norm
from test_super_blocking import get_super_blocking_keys

# Load validation S1 set
val_s1_ids = []
with open('experiments/val_s1_ids.txt', 'r', encoding='utf-8') as f:
    for line in f:
        val_s1_ids.append(line.strip())
        if len(val_s1_ids) >= 5000:
            break

val_s1_set = set(val_s1_ids)

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

# Load S1 raw
s1_raw = {}
with open('student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in val_s1_set:
            s1_raw[p[0]] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else '')

# Load Targets (True targets + 40,000 distractors)
target_raw = {}
dist_count = 0
with open('student_resource/dataset/train/train_source2.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        tid = p[0]
        if tid in val_true_targets or dist_count < 25000:
            target_raw[tid] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else '')
            if tid not in val_true_targets:
                dist_count += 1

dist_count3 = 0
with open('student_resource/dataset/train/train_source3.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        tid = p[0]
        if tid in val_true_targets or dist_count3 < 25000:
            target_raw[tid] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else '')
            if tid not in val_true_targets:
                dist_count3 += 1

print(f'Target pool: {len(target_raw):,} records')

# Build index: do not prune name-based keys as aggressively
index = collections.defaultdict(lambda: collections.defaultdict(list))
for tid, (rname, raddr, rcountry) in target_raw.items():
    keys = get_super_blocking_keys(rname, raddr, rcountry)
    for k in keys:
        index[rcountry][k].append(tid)

# Only prune address pairs that have > 200 occurrences; keep name keys up to 300
for c in index:
    for k in list(index[c].keys()):
        limit = 300 if k[0].startswith('n') or k[0].startswith('core') or k[0].startswith('compact') else 150
        if len(index[c][k]) > limit:
            del index[c][k]

total_true = sum(len(v) for v in val_gt.values())
retrieved = 0
total_cands = 0

for sid in val_s1_ids:
    rname, raddr, rcountry = s1_raw[sid]
    s_keys = get_super_blocking_keys(rname, raddr, rcountry)
    c_index = index[rcountry]
    counts = collections.Counter()
    for k in s_keys:
        if k in c_index:
            counts.update(c_index[k])
    cands = [tid for tid, _ in counts.most_common(20)]
    retrieved += len(val_gt[sid] & set(cands))
    total_cands += len(cands)

recall = retrieved / total_true
print(f'Tuned Pruning (Name<=300, Addr<=150): Candidate Recall={retrieved}/{total_true} = {recall*100:.2f}%, Avg Cands={total_cands/len(val_s1_ids):.2f}')
