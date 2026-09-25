import sys
import os
import re
import collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution/src')
import normalization as norm
from test_blocking_recall import s1_data, target_data, val_gt

COMMON_ADDR_WORDS = {
    'street', 'road', 'avenue', 'boulevard', 'drive', 'court', 'lane', 'place',
    'circle', 'way', 'trail', 'parkway', 'highway', 'suite', 'floor', 'apartment',
    'building', 'room', 'number', 'near', 'opposite', 'behind', 'block', 'sector',
    'phase', 'plot', 'flat', 'door', 'fl', 'no', 'unit', 'north', 'south', 'east', 'west',
    'india', 'us', 'usa', 'france', 'state', 'district', 'city', 'nagar', 'colony',
    'bazaar', 'marg', 'road', 'gali', 'null'
}

def get_enhanced_blocking_keys(name, addr, country):
    c_n, core_n, sort_n = norm.normalize_name(name)
    c_a, nums, sort_a = norm.normalize_address(addr)
    
    keys = set()
    n_tokens = core_n.split()
    a_tokens = c_a.split()
    
    # 1. Compact name (no spaces) if length >= 5
    compact_name = ''.join(n_tokens)
    if len(compact_name) >= 5:
        keys.add(('compact_n', compact_name))
        
    # 2. Core name exact
    if len(core_n) >= 3:
        keys.add(('core_n', core_n))
        
    # 3. First 2 tokens of core name
    if len(n_tokens) >= 2:
        keys.add(('n2', f'{n_tokens[0]}_{n_tokens[1]}'))
        keys.add(('sort_n', ' '.join(sorted(n_tokens))))
    elif len(n_tokens) == 1 and len(n_tokens[0]) >= 4:
        keys.add(('n1', n_tokens[0]))

    # 4. Individual rare name tokens (length >= 5)
    for t in n_tokens:
        if len(t) >= 6:
            keys.add(('n_rare', t))
            
    # Address tokens: extract significant/rare words
    sig_addr_words = [t for t in a_tokens if t not in COMMON_ADDR_WORDS and len(t) >= 4 and not t.isdigit()]
    
    # 5. Number + City (last 1 or 2 tokens of address)
    if nums and len(a_tokens) >= 1:
        for num in nums:
            if len(num) >= 2:
                keys.add(('num_city1', f'{num}_{a_tokens[-1]}'))
                if len(a_tokens) >= 2:
                    keys.add(('num_city2', f'{num}_{a_tokens[-2]}'))
                    
    # 6. Street number + rare address word
    if nums and sig_addr_words:
        for num in nums:
            if len(num) >= 2:
                for w in sig_addr_words[:2]:
                    keys.add(('num_street', f'{num}_{w}'))
                    
    # 7. Name token + Street number
    if n_tokens and nums:
        for num in nums:
            if len(num) >= 2:
                keys.add(('name_num', f'{n_tokens[0]}_{num}'))
                
    # 8. Name token + City
    if n_tokens and len(a_tokens) >= 1:
        keys.add(('name_city', f'{n_tokens[0]}_{a_tokens[-1]}'))
        if len(a_tokens) >= 2:
            keys.add(('name_city2', f'{n_tokens[0]}_{a_tokens[-2]}'))
            
    # 9. Top 2 rare address words combined
    if len(sig_addr_words) >= 2:
        keys.add(('addr_pair', f'{sig_addr_words[0]}_{sig_addr_words[1]}'))
        
    return keys

# Evaluate enhanced keys
total_links = sum(len(mids) for mids in val_gt.values())
key_success = collections.Counter()
any_success = 0

for s1_id, mids in val_gt.items():
    s1_name, s1_addr, s1_c = s1_data[s1_id]
    s1_keys = get_enhanced_blocking_keys(s1_name, s1_addr, s1_c)
    
    for mid in mids:
        if mid not in target_data:
            continue
        t_name, t_addr, t_c = target_data[mid]
        t_keys = get_enhanced_blocking_keys(t_name, t_addr, t_c)
        
        shared = s1_keys & t_keys
        if shared:
            any_success += 1
            for k in shared:
                key_success[k[0]] += 1

print(f'Total positive links: {total_links}')
print(f'Enhanced blocking recall: {any_success} / {total_links} = {any_success/total_links*100:.2f}% recall!')
print('Recall by key type:')
for k, cnt in key_success.most_common():
    print(f'  {k}: {cnt} / {total_links} ({cnt/total_links*100:.2f}%)')
