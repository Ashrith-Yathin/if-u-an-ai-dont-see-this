import sys
import os
import re
import collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution/src')
import normalization as norm

# Load sample validation entities
val_ids = set()
with open('experiments/val_s1_ids.txt', 'r', encoding='utf-8') as f:
    for line in f:
        val_ids.add(line.strip())
        if len(val_ids) >= 2000:  # test on 2000 val entities first
            break

# Load S1 data for val_ids
s1_data = {}
with open('student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in val_ids:
            s1_data[p[0]] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')

# Load ground truth for val_ids
val_gt = {}
needed_targets = set()
with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in val_ids:
            mids = p[1].split(',') if len(p)>1 and p[1] else []
            val_gt[p[0]] = set(mids)
            needed_targets.update(mids)

# Load S2 and S3 for needed_targets
target_data = {}
with open('student_resource/dataset/train/train_source2.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in needed_targets:
            target_data[p[0]] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')

with open('student_resource/dataset/train/train_source3.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in needed_targets:
            target_data[p[0]] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')

print(f'Loaded {len(s1_data)} S1 records, {len(target_data)} true target records.')

# Test blocking keys
def get_blocking_keys(name, addr, country):
    c_n, core_n, sort_n = norm.normalize_name(name)
    c_a, nums, sort_a = norm.normalize_address(addr)
    
    keys = set()
    n_tokens = core_n.split()
    a_tokens = c_a.split()
    
    # Key 1: exact core name
    if len(core_n) >= 3:
        keys.add(('core_n', core_n))
    
    # Key 2: first 2 tokens of core name
    if len(n_tokens) >= 2:
        keys.add(('n2', f'{n_tokens[0]}_{n_tokens[1]}'))
    elif len(n_tokens) == 1 and len(n_tokens[0]) >= 3:
        keys.add(('n1', n_tokens[0]))
        
    # Key 3: token sorted core name
    if len(n_tokens) >= 2:
        keys.add(('sort_n', ' '.join(sorted(n_tokens))))
        
    # Key 4: street number + first word of street/address
    if nums and len(a_tokens) >= 2:
        first_num = sorted(list(nums), key=lambda x: len(x), reverse=True)[0]
        # find token after number or main street token
        non_num_tokens = [t for t in a_tokens if t not in nums and len(t) >= 3]
        if non_num_tokens:
            keys.add(('num_street', f'{first_num}_{non_num_tokens[0]}'))
            if len(non_num_tokens) >= 2:
                keys.add(('num_street2', f'{first_num}_{non_num_tokens[1]}'))
                
    # Key 5: first core name token + street number
    if n_tokens and nums:
        first_num = sorted(list(nums), key=lambda x: len(x), reverse=True)[0]
        keys.add(('name_num', f'{n_tokens[0]}_{first_num}'))
        
    # Key 6: first core name token + state/city (last 2 tokens of address)
    if n_tokens and len(a_tokens) >= 2:
        keys.add(('name_city', f'{n_tokens[0]}_{a_tokens[-1]}'))
        keys.add(('name_state', f'{n_tokens[0]}_{a_tokens[-2]}'))
        
    # Key 7: exact normalized address
    if len(c_a) >= 10:
        keys.add(('addr', c_a))
        
    return keys

# Evaluate key recall
total_links = sum(len(mids) for mids in val_gt.values())
key_success = collections.Counter()
any_success = 0

for s1_id, mids in val_gt.items():
    s1_name, s1_addr, s1_c = s1_data[s1_id]
    s1_keys = get_blocking_keys(s1_name, s1_addr, s1_c)
    
    for mid in mids:
        if mid not in target_data:
            continue
        t_name, t_addr, t_c = target_data[mid]
        t_keys = get_blocking_keys(t_name, t_addr, t_c)
        
        shared = s1_keys & t_keys
        if shared:
            any_success += 1
            for k in shared:
                key_success[k[0]] += 1

print(f'Total positive links: {total_links}')
print(f'Links captured by blocking keys: {any_success} / {total_links} = {any_success/total_links*100:.2f}% recall!')
print('Recall by key type:')
for k, cnt in key_success.most_common():
    print(f'  {k}: {cnt} / {total_links} ({cnt/total_links*100:.2f}%)')
