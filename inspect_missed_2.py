import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution/src')
import normalization as norm
from test_blocking_recall import s1_data, target_data, val_gt
from test_enhanced_blocking import get_enhanced_blocking_keys

missed = []
for s1_id, mids in val_gt.items():
    s1_name, s1_addr, s1_c = s1_data[s1_id]
    s1_keys = get_enhanced_blocking_keys(s1_name, s1_addr, s1_c)
    for mid in mids:
        if mid not in target_data:
            continue
        t_name, t_addr, t_c = target_data[mid]
        t_keys = get_enhanced_blocking_keys(t_name, t_addr, t_c)
        shared = s1_keys & t_keys
        if not shared:
            missed.append((s1_id, mid, s1_name, t_name, s1_addr, t_addr, s1_c))
            if len(missed) >= 15:
                break
    if len(missed) >= 15:
        break

print(f'Total remaining missed examples shown ({len(missed)}):')
for m in missed:
    print(f'S1: {m[0]} | T: {m[1]} ({m[6]})')
    print(f'  NAME S1: {m[2]} | T: {m[3]}')
    print(f'  ADDR S1: {m[4]} | T: {m[5]}')
    print()
