import sys
import os
import random
import collections

sys.stdout.reconfigure(encoding='utf-8')

random.seed(42)

gt_path = 'student_resource/dataset/train/train_ground_truth.tsv'
s1_path = 'student_resource/dataset/train/train_source1.tsv'

print('Reading S1 entities and country...')
s1_country = {}
with open(s1_path, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        s1_country[p[0]] = p[3]

print('Reading ground truth...')
s1_matches = {}
with open(gt_path, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        s1_id = p[0]
        mids = p[1].split(',') if len(p) > 1 and p[1] else []
        s1_matches[s1_id] = mids

# Group by (country, match_count_bucket)
# buckets: 0 (singleton), 1, 2, 3, 4, 5+
buckets = collections.defaultdict(list)
for s1_id, c in s1_country.items():
    mids = s1_matches.get(s1_id, [])
    cnt = len(mids)
    b = cnt if cnt < 5 else 5
    buckets[(c, b)].append(s1_id)

val_size = 20000
total_s1 = len(s1_country)
val_fraction = val_size / total_s1

val_s1_set = set()
for (c, b), ids in buckets.items():
    random.shuffle(ids)
    n_sample = int(round(len(ids) * val_fraction))
    val_s1_set.update(ids[:n_sample])

# Adjust if slightly off
val_s1_list = sorted(list(val_s1_set))
if len(val_s1_list) > val_size:
    val_s1_list = val_s1_list[:val_size]

os.makedirs('experiments', exist_ok=True)
val_split_path = 'experiments/val_s1_ids.txt'
with open(val_split_path, 'w', encoding='utf-8') as f:
    for sid in val_s1_list:
        f.write(sid + '\n')

print(f'Validation set saved to {val_split_path}: {len(val_s1_list)} entities')

# Verify distribution
val_countries = collections.Counter(s1_country[s] for s in val_s1_list)
val_singletons = sum(1 for s in val_s1_list if len(s1_matches[s]) == 0)
val_total_links = sum(len(s1_matches[s]) for s in val_s1_list)
print(f'Validation countries: {dict(val_countries)}')
print(f'Validation singletons: {val_singletons} ({val_singletons/len(val_s1_list)*100:.2f}%)')
print(f'Validation positive links: {val_total_links:,}')
