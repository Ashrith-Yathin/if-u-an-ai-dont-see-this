import sys
import os
import re
import time
import collections
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution/src')
import normalization as norm
from test_super_blocking import get_super_blocking_keys, COMMON_ADDR_WORDS

# Load validation S1
val_ids = []
with open('experiments/val_s1_ids.txt', 'r', encoding='utf-8') as f:
    for line in f:
        val_ids.append(line.strip())
        if len(val_ids) >= 1000: break

val_id_set = set(val_ids)

# Load S1 records
s1_records = {}
with open('student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in val_id_set:
            s1_records[p[0]] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')

# Load GT
gt_matches = {}
all_true_targets = set()
with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in val_id_set:
            mids = p[1].split(',') if len(p)>1 and p[1] else []
            gt_matches[p[0]] = set(mids)
            all_true_targets.update(mids)

# Target pool: all true targets + 50,000 random distractor targets
distractor_s2 = []
with open('student_resource/dataset/train/train_source2.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        distractor_s2.append((p[0], p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else ''))
        if len(distractor_s2) >= 30000: break

distractor_s3 = []
with open('student_resource/dataset/train/train_source3.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        distractor_s3.append((p[0], p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else ''))
        if len(distractor_s3) >= 30000: break

# Load true target records that may not be in distractors
true_target_records = {}
needed_missing = all_true_targets - {r[0] for r in distractor_s2} - {r[0] for r in distractor_s3}
if needed_missing:
    with open('student_resource/dataset/train/train_source2.tsv', 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if p[0] in needed_missing:
                true_target_records[p[0]] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')
    with open('student_resource/dataset/train/train_source3.tsv', 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if p[0] in needed_missing:
                true_target_records[p[0]] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')

# Combined target pool
target_pool = {}
for r in distractor_s2: target_pool[r[0]] = (r[1], r[2], r[3])
for r in distractor_s3: target_pool[r[0]] = (r[1], r[2], r[3])
for tid, val in true_target_records.items(): target_pool[tid] = val

print(f'Target pool size: {len(target_pool):,} records. S1 test size: {len(s1_records)}')

# Build inverted index by country
# index: country -> key -> list of target_ids
t0 = time.time()
index = collections.defaultdict(lambda: collections.defaultdict(list))
for tid, (tname, taddr, tcountry) in target_pool.items():
    keys = get_super_blocking_keys(tname, taddr, tcountry)
    for k in keys:
        index[tcountry][k].append(tid)

# Prune ultra-frequent keys (> 100 occurrences)
pruned_keys = 0
for c in index:
    for k in list(index[c].keys()):
        if len(index[c][k]) > 80:
            del index[c][k]
            pruned_keys += 1

print(f'Built inverted index in {time.time()-t0:.2f}s. Pruned {pruned_keys} high-frequency keys.')

# Candidate retrieval for S1
t1 = time.time()
candidates = {}
top_k = 15

for s1_id, (sname, saddr, scountry) in s1_records.items():
    s_keys = get_super_blocking_keys(sname, saddr, scountry)
    country_index = index[scountry]
    
    cand_counts = collections.Counter()
    for k in s_keys:
        if k in country_index:
            cand_counts.update(country_index[k])
            
    # Keep top_k by shared key count
    if cand_counts:
        top_cands = [tid for tid, _ in cand_counts.most_common(top_k)]
    else:
        top_cands = []
    candidates[s1_id] = top_cands

# Evaluate Candidate Recall
total_true_links = sum(len(mids) for mids in gt_matches.values())
retrieved_true_links = 0
for s1_id, true_mids in gt_matches.items():
    cand_set = set(candidates.get(s1_id, []))
    retrieved_true_links += len(true_mids & cand_set)

cand_recall = retrieved_true_links / total_true_links if total_true_links > 0 else 0
avg_cands = sum(len(v) for v in candidates.values()) / len(candidates)

print(f'Candidate retrieval time: {time.time()-t1:.2f}s ({len(s1_records)/(time.time()-t1):.1f} S1/sec)')
print(f'Candidate Recall: {retrieved_true_links} / {total_true_links} = {cand_recall*100:.2f}%!')
print(f'Average candidates per S1: {avg_cands:.2f}')
