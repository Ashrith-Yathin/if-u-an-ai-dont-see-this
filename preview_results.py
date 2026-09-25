import sys

sys.stdout.reconfigure(encoding='utf-8')

matching_path = 'output/matching_results.tsv'
candidate_path = 'output/candidate_pairs.tsv'

print('============================================================')
print('PREVIEW OF matching_results.tsv (First 20 rows):')
print('============================================================')
with open(matching_path, 'r', encoding='utf-8') as f:
    for i in range(21):
        line = f.readline()
        if not line:
            break
        print(f'{i:2d}: {line.rstrip()}')

print('\n============================================================')
print('PREVIEW OF candidate_pairs.tsv (First 20 rows):')
print('============================================================')
with open(candidate_path, 'r', encoding='utf-8') as f:
    for i in range(21):
        line = f.readline()
        if not line:
            break
        print(f'{i:2d}: {line.rstrip()}')

# Find sample rows with singletons and multiple matches
singletons = []
multi_matches = []
with open(matching_path, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        line = line.rstrip('\r\n')
        parts = line.split('\t')
        s1 = parts[0]
        mids = parts[1] if len(parts) > 1 else ''
        if not mids and len(singletons) < 5:
            singletons.append(s1)
        elif mids and ',' in mids and len(multi_matches) < 5:
            multi_matches.append((s1, mids))
        if len(singletons) >= 5 and len(multi_matches) >= 5:
            break

print('\n============================================================')
print('SAMPLE SINGLETON PREDICTIONS (Empty matched_entity_ids):')
print('============================================================')
for s in singletons:
    print(f'{s}\t')

print('\n============================================================')
print('SAMPLE MULTIPLE MATCH PREDICTIONS:')
print('============================================================')
for s, mids in multi_matches:
    print(f'{s}\t{mids}')
