import numpy as np
import collections
from evaluation import evaluate_predictions


def optimize_global_threshold(ground_truth_dict, candidate_scores_dict, thresholds=None):
    """
    candidate_scores_dict: {s1_id: [(target_id, proba), ...]}
    Finds the threshold that maximizes macro F_0.5.
    """
    if thresholds is None:
        thresholds = np.linspace(0.30, 0.90, 31)

    best_threshold = 0.5
    best_metrics = None
    best_f05 = -1.0

    for thresh in thresholds:
        preds = {}
        for s1_id, scores in candidate_scores_dict.items():
            matched = {tid for tid, p in scores if p >= thresh}
            preds[s1_id] = matched

        metrics = evaluate_predictions(ground_truth_dict, preds)
        if metrics['macro_f05'] > best_f05:
            best_f05 = metrics['macro_f05']
            best_threshold = float(thresh)
            best_metrics = metrics

    return best_threshold, best_metrics


def _sweep_threshold_numpy(val_gt, scores_dict, subset_pairs):
    s1_ids = {sid for sid, tid in subset_pairs}
    local_gt = {k: v for k, v in val_gt.items() if k in s1_ids}
    subset_set = set(subset_pairs)
    
    best_th, best_f = 0.5, -1
    for th in np.arange(0.1, 0.96, 0.02):
        preds = collections.defaultdict(set)
        for sid in s1_ids:
            for tid, p in scores_dict.get(sid, []):
                if (sid, tid) in subset_set and p >= th:
                    preds[sid].add(tid)
                    
        metrics = evaluate_predictions(local_gt, preds)
        if metrics['macro_f05'] > best_f:
            best_f = metrics['macro_f05']
            best_th = float(th)
            
    return best_th, best_f


def calibrate_thresholds_2d(val_gt, scores_dict, val_pair_list, val_countries, min_group_size=1000):
    subset_all = [(sid, tid) for sid, tid, _ in val_pair_list]
    global_th, global_f = _sweep_threshold_numpy(val_gt, scores_dict, subset_all)
    print(f'GLOBAL: threshold={global_th:.2f} F_0.5={global_f:.4f} (n={len(subset_all)})')
    
    country_th = {}
    country_pairs = collections.defaultdict(list)
    for i, (sid, tid, _) in enumerate(val_pair_list):
        c = val_countries[i]
        country_pairs[c].append((sid, tid))
        
    for country, pairs in country_pairs.items():
        if len(pairs) < min_group_size:
            country_th[country] = global_th
            continue
        th, f = _sweep_threshold_numpy(val_gt, scores_dict, pairs)
        country_th[country] = th
        print(f'{country}: threshold={th:.2f} F_0.5={f:.4f} (n={len(pairs)})')
        
    country_source_th = {}
    country_source_pairs = collections.defaultdict(list)
    for i, (sid, tid, _) in enumerate(val_pair_list):
        c = val_countries[i]
        source = tid[:2]
        country_source_pairs[(c, source)].append((sid, tid))
        
    for (country, source), pairs in country_source_pairs.items():
        if len(pairs) < min_group_size:
            th_fallback = country_th.get(country, global_th)
            country_source_th[f"{country}_{source}"] = th_fallback
            print(f'{country}/{source}: n={len(pairs)} < {min_group_size} -> fallback to {th_fallback:.2f}')
            continue
        th, f = _sweep_threshold_numpy(val_gt, scores_dict, pairs)
        country_source_th[f"{country}_{source}"] = th
        print(f'{country}/{source}: threshold={th:.2f} F_0.5={f:.4f} (n={len(pairs)})')
        
    return {'global': global_th, 'country': country_th, 'country_source': country_source_th}


def lookup_threshold(threshold_map_2d, country, source):
    if not threshold_map_2d:
        return 0.5
    cs_map = threshold_map_2d.get('country_source', {})
    c_map = threshold_map_2d.get('country', {})
    g_th = threshold_map_2d.get('global', 0.5)
    
    cs_key = f"{country}_{source}"
    if cs_key in cs_map:
        return cs_map[cs_key]
    if country in c_map:
        return c_map[country]
    return g_th


def apply_threshold_and_deduplication(candidate_scores_dict, s2_threshold=0.5, s3_threshold=0.5, threshold_map_2d=None, s1_countries_map=None):
    """
    Applies thresholds and enforces that each target (S2/S3) is assigned to at most one S1
    (the S1 with the highest probability score).
    """
    all_pairs = []
    for s1_id, scores in candidate_scores_dict.items():
        c = s1_countries_map.get(s1_id, 'us') if s1_countries_map else 'us'
            
        for tid, p in scores:
            if threshold_map_2d:
                t = lookup_threshold(threshold_map_2d, c, tid[:2])
            else:
                t = s2_threshold if tid.startswith('S2-') else s3_threshold
                
            if p >= t:
                all_pairs.append((p, s1_id, tid))

    # Sort descending by score
    all_pairs.sort(key=lambda x: x[0], reverse=True)

    assigned_targets = set()
    result = {s1_id: set() for s1_id in candidate_scores_dict.keys()}

    for p, s1_id, tid in all_pairs:
        if tid not in assigned_targets:
            assigned_targets.add(tid)
            result[s1_id].add(tid)

    return result
