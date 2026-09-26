"""
End-to-end test-set inference using Country-by-Country processing.

KEY ARCHITECTURAL CHANGE (v2 - inspired by Akash-bardia 0.97+ approach):
Instead of generating ALL 16M+ candidate pairs up-front and scoring them
in one giant loop, we now process each country separately:
  1. Load S2/S3 records for ONLY that country
  2. Build multi-key inverted index for that country
  3. Generate candidates, score them
  4. DELETE everything and move to the next country

Peak RAM: ~4-6GB (vs 30GB+ for old approach)
Runtime: ~45-60 minutes on Kaggle (vs 1hr old approach, at much better recall)

Usage:
    python inference_pipeline.py
    python inference_pipeline.py --threshold 0.70
    python inference_pipeline.py --s2_threshold 0.75 --s3_threshold 0.85
"""

import argparse
import os
import gc
import collections
import numpy as np
import pandas as pd
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import torch

from config import CFG
from data_utils import load_source, write_matching_results, write_candidate_pairs
from blocking import get_blocking_keys
from features import df_to_lookup, process_chunk
from train_matcher import MLP, get_mlp_preds


def _fit_tfidf_memory_efficient(*paths) -> TfidfVectorizer:
    """Fits TF-IDF by streaming text from each source without holding all
    DataFrames in memory simultaneously."""
    print("  Fitting TF-IDF (memory-efficient streaming across all sources)...")
    all_text = []
    for path in paths:
        print(f"    collecting text from {os.path.basename(path)}...")
        df = load_source(path)
        if len(df) > CFG.TFIDF_SAMPLE_SIZE:
            df = df.sample(n=CFG.TFIDF_SAMPLE_SIZE, random_state=42)
        all_text.extend((df["norm_name"] + " " + df["norm_addr"]).tolist())
        del df
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20000)
    vec.fit(all_text)
    del all_text
    print("  TF-IDF fitted.")
    return vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold_file", default="outputs_threshold.txt")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Single threshold for both S2 and S3")
    ap.add_argument("--s2_threshold", type=float, default=None,
                    help="Source-2 specific threshold (default: use --threshold)")
    ap.add_argument("--s3_threshold", type=float, default=None,
                    help="Source-3 specific threshold (default: use --threshold)")
    ap.add_argument("--top_k", type=int, default=15,
                    help="Max candidates per S1 entity from multi-key index")
    args = ap.parse_args()

    # Determine thresholds
    global_threshold = args.threshold
    if global_threshold is None:
        with open(args.threshold_file) as f:
            global_threshold = float(f.read().strip())
    s2_threshold = args.s2_threshold if args.s2_threshold is not None else global_threshold
    s3_threshold = args.s3_threshold if args.s3_threshold is not None else global_threshold
    print(f"Decision thresholds: S2={s2_threshold:.3f}, S3={s3_threshold:.3f}, top_k={args.top_k}")

    # Fit TF-IDF once across all sources
    print("\nFitting TF-IDF across all sources (train + test)...")
    tfidf_vec = _fit_tfidf_memory_efficient(
        CFG.TRAIN_S1, CFG.TRAIN_S2, CFG.TRAIN_S3,
        CFG.TEST_S1,  CFG.TEST_S2,  CFG.TEST_S3,
    )

    # Load models
    print("\nLoading Blended Models (XGBoost + PyTorch MLP)...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("outputs_xgb_model.json")
    mlp_stats = np.load("outputs_mlp_stats.npy", allow_pickle=True).item()
    mlp_mean, mlp_std = mlp_stats["mean"], mlp_stats["std"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_model = MLP(14).to(device)
    mlp_model.load_state_dict(torch.load("outputs_mlp_model.pth", map_location=device))
    mlp_model.eval()
    print(f"  Models loaded. Device: {device}")

    # Load S1 (reference), S2, S3
    print("\nLoading test sources...")
    s1 = load_source(CFG.TEST_S1)
    if CFG.TEST_ENTITY_SAMPLE and CFG.TEST_ENTITY_SAMPLE < len(s1):
        s1 = s1.sample(n=CFG.TEST_ENTITY_SAMPLE, random_state=42).reset_index(drop=True)
    s2 = load_source(CFG.TEST_S2)
    s3 = load_source(CFG.TEST_S3)
    other = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    ordered_s1_ids = list(s1["entity_id"])
    all_s1_ids = set(ordered_s1_ids)
    countries = sorted(s1["country"].unique())
    print(f"Countries in test set: {countries}")
    for c in countries:
        print(f"  {c}: {(s1['country']==c).sum():,} S1 entities")

    # Result containers
    all_matches = {sid: set() for sid in all_s1_ids}
    all_candidates = {sid: set() for sid in all_s1_ids}

    # COUNTRY-BY-COUNTRY PROCESSING — the key to no OOM!
    for country in countries:
        s1_country = s1[s1["country"] == country].copy()
        other_country = other[other["country"] == country].copy()
        print(f"\n{'='*60}")
        print(f"Processing: {country} | {len(s1_country):,} S1 | {len(other_country):,} S2+S3 targets")

        # Build multi-key inverted index for this country only
        print(f"  Building inverted index...")
        index = collections.defaultdict(list)
        for row in tqdm(other_country.itertuples(index=False),
                        total=len(other_country), desc="  Indexing targets", mininterval=5.0):
            keys = get_blocking_keys(row.norm_name, row.norm_addr, row.country)
            for k in keys:
                index[k].append(row.entity_id)

        # Prune ultra-frequent keys (generic buckets like "the" that match everything)
        pruned = sum(1 for k in list(index.keys()) if len(index[k]) > 300)
        for k in list(index.keys()):
            if len(index[k]) > 300:
                del index[k]
        print(f"  Index: {len(index):,} active keys (pruned {pruned:,} overly-common keys)")

        # Retrieve top_k candidates for each S1 entity
        country_candidates = {}
        for row in tqdm(s1_country.itertuples(index=False),
                        total=len(s1_country), desc=f"  Retrieving candidates", mininterval=5.0):
            s_keys = get_blocking_keys(row.norm_name, row.norm_addr, row.country)
            counts = collections.Counter()
            for k in s_keys:
                if k in index:
                    counts.update(index[k])
            if counts:
                cands = [cid for cid, _ in counts.most_common(args.top_k)]
                country_candidates[row.entity_id] = cands
                all_candidates[row.entity_id].update(cands)

        del index
        gc.collect()

        # Score all candidate pairs for this country
        pairs = [(s1_id, cid) for s1_id, cids in country_candidates.items() for cid in cids]
        if not pairs:
            print(f"  No candidates found for {country}.")
            del other_country, country_candidates
            gc.collect()
            continue

        print(f"  Scoring {len(pairs):,} candidate pairs...")
        s1_lookup = df_to_lookup(s1_country)
        other_lookup = df_to_lookup(other_country)

        CHUNK = 100_000
        for i in range(0, len(pairs), CHUNK):
            chunk = pairs[i:i + CHUNK]
            X, valid_pairs = process_chunk(chunk, s1_lookup, other_lookup, tfidf_vec)
            if not len(X):
                continue

            xgb_preds = xgb_model.predict_proba(X)[:, 1]
            mlp_preds = get_mlp_preds(mlp_model, X, mlp_mean, mlp_std)
            scores = (xgb_preds + mlp_preds) / 2.0

            for (s1_id, other_id), score in zip(valid_pairs, scores):
                thresh = s2_threshold if other_id.startswith("S2-") else s3_threshold
                if score >= thresh:
                    all_matches[s1_id].add(other_id)

        matched_in_country = sum(1 for sid in s1_country["entity_id"] if all_matches[sid])
        print(f"  Done. {matched_in_country:,}/{len(s1_country):,} S1 entities matched in {country}.")

        del other_country, country_candidates, pairs, s1_lookup, other_lookup
        gc.collect()

    # Write outputs
    os.makedirs(CFG.OUT_DIR, exist_ok=True)
    write_candidate_pairs(all_candidates)
    print(f"\nWrote {CFG.CANDIDATES_OUT} ({sum(len(v) for v in all_candidates.values()):,} candidate pairs)")
    write_matching_results(all_matches)
    n_matched = sum(1 for v in all_matches.values() if v)
    print(f"Wrote {CFG.MATCHING_OUT} ({n_matched:,}/{len(all_matches):,} S1 entities have >=1 match)")


if __name__ == "__main__":
    main()
