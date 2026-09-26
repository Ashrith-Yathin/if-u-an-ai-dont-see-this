"""
Candidate generation (blocking). This determines your recall ceiling — the
PS explicitly calls this out as the top priority. Two complementary signals,
unioned together:

  1. String-key blocking: exact match on (country, normalized-name-prefix).
     Cheap, high-precision, but brittle to typos/transliteration/word-order.
  2. Embedding kNN blocking: cosine nearest-neighbours in sentence-embedding
     space over (name + address). Catches paraphrase-level variation the
     string key misses, at the cost of being slower.

Returns candidates as {s1_entity_id: set(s2/s3_entity_ids)}.
"""

import os
import gc
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CFG
from normalize import blocking_key


def save_candidates_cache(candidates: dict, cache_path: str):
    """Saves candidate mapping {s1_id: set(cand_ids)} to compressed parquet."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    s1_ids = list(candidates.keys())
    cand_strs = [",".join(sorted(candidates[k])) if candidates[k] else "" for k in s1_ids]
    df = pd.DataFrame({"source1_entity_id": s1_ids, "candidate_ids": cand_strs})
    df.to_parquet(cache_path, index=False)
    print(f"  Saved candidate cache to {cache_path} ({len(candidates):,} entities)")


def load_candidates_cache(cache_path: str) -> dict:
    """Loads candidate mapping from parquet cache."""
    print(f"  Loading candidate cache from {cache_path}...")
    df = pd.read_parquet(cache_path)
    s1_ids = df["source1_entity_id"].values
    cand_strs = df["candidate_ids"].values
    loaded = {}
    for s1_id, c_str in zip(s1_ids, cand_strs):
        loaded[s1_id] = set(c_str.split(",")) if c_str else set()
    return loaded


import re
from collections import Counter

COMMON_ADDR_WORDS = {
    'street', 'road', 'avenue', 'boulevard', 'drive', 'court', 'lane', 'place',
    'circle', 'way', 'trail', 'parkway', 'highway', 'suite', 'floor', 'apartment',
    'building', 'room', 'number', 'near', 'opposite', 'behind', 'block', 'sector',
    'phase', 'plot', 'flat', 'door', 'fl', 'no', 'unit', 'north', 'south', 'east', 'west',
    'india', 'us', 'usa', 'france', 'state', 'district', 'city', 'nagar', 'colony',
    'bazaar', 'marg', 'gali', 'null', 'rd', 'st', 'ave', 'blvd', 'dr', 'ct', 'ln',
    'hwy', 'apt', 'ste', 'first', 'second', 'third', 'ground'
}

def get_blocking_keys(norm_name, norm_addr, country):
    keys = set()
    if not isinstance(norm_name, str): norm_name = ""
    if not isinstance(norm_addr, str): norm_addr = ""
    
    n_tokens = norm_name.split()
    a_tokens = norm_addr.split()
    nums = set(re.findall(r'\b\d+\b', norm_addr))
    
    # 1. Compact name
    compact_name = ''.join(n_tokens)
    if len(compact_name) >= 4:
        keys.add((country, 'compact_n', compact_name))
        
    # 2. Core name exact
    if len(norm_name) >= 3:
        keys.add((country, 'core_n', norm_name))
        
    # 3. First 2 tokens of name
    if len(n_tokens) >= 2:
        keys.add((country, 'n2', f'{n_tokens[0]}_{n_tokens[1]}'))
    elif len(n_tokens) == 1 and len(n_tokens[0]) >= 3:
        keys.add((country, 'n1', n_tokens[0]))
        
    # 4. Individual name tokens
    for t in n_tokens:
        if len(t) >= 4:
            keys.add((country, 'n_tok', t))
            
    # Address tokens: extract significant words
    sig_addr_words = [t for t in a_tokens if t not in COMMON_ADDR_WORDS and len(t) >= 3 and not t.isdigit()]
    
    # 5. Number + City (last 2 tokens of address)
    if nums and len(a_tokens) >= 1:
        for num in nums:
            keys.add((country, 'num_city1', f'{num}_{a_tokens[-1]}'))
            if len(a_tokens) >= 2:
                keys.add((country, 'num_city2', f'{num}_{a_tokens[-2]}'))
                
    # 6. Street number + rare address word
    if nums and sig_addr_words:
        for num in nums:
            for w in sig_addr_words[:3]:
                keys.add((country, 'num_street', f'{num}_{w}'))
                
    # 7. Name token + Street number
    if n_tokens and nums:
        for num in nums:
            keys.add((country, 'name_num', f'{n_tokens[0]}_{num}'))
            if len(n_tokens) >= 2:
                keys.add((country, 'name_num2', f'{n_tokens[1]}_{num}'))
                
    return keys

def string_key_candidates(s1_df, other_df, prefix_len=None, max_candidates=None, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        return load_candidates_cache(cache_path)

    # 1. Build Multi-Key Inverted Index
    index = defaultdict(list)
    for row in tqdm(other_df.itertuples(index=False), total=len(other_df), desc="  Indexing multi-keys (target)", mininterval=5.0):
        keys = get_blocking_keys(row.norm_name, row.norm_addr, row.country)
        for k in keys:
            index[k].append(row.entity_id)

    # Prune ultra-frequent keys to prevent generic matches (e.g. matching 10,000 businesses in one key)
    prune_limit = 300
    for k in list(index.keys()):
        if len(index[k]) > prune_limit:
            del index[k]

    # 2. Retrieve Candidates
    candidates = {}
    # We aim for ~15 high-quality candidates per entity
    top_k_candidates = max_candidates or getattr(CFG, "MAX_STRING_CANDIDATES_PER_KEY", 15)
    
    for row in tqdm(s1_df.itertuples(index=False), total=len(s1_df), desc="  Retrieving multi-key candidates (source)", mininterval=5.0):
        s_keys = get_blocking_keys(row.norm_name, row.norm_addr, row.country)
        counts = Counter()
        for k in s_keys:
            if k in index:
                counts.update(index[k])
                
        if counts:
            # Only keep the candidates that share the most specific keys
            candidates[row.entity_id] = [cid for cid, count in counts.most_common(top_k_candidates)]

    del index
    gc.collect()

    if cache_path:
        save_candidates_cache(candidates, cache_path)

    return candidates


def _combined_text(df):
    return (df["norm_name"] + " " + df["norm_addr"]).tolist()


def _get_or_compute_embeddings(df, country, model, batch_size, cache_key=None):
    """
    Computes or loads normalized dense embeddings (np.float32).
    If CFG.SAVE_RAW_EMBEDDINGS is True, saves/loads vectors from disk in FP16 to save space.
    """
    cache_file = None
    save_raw = getattr(CFG, "SAVE_RAW_EMBEDDINGS", True)
    if save_raw and cache_key:
        vec_dir = os.path.join(CFG.CACHE_DIR, "vectors")
        os.makedirs(vec_dir, exist_ok=True)
        safe_country = "".join(c if c.isalnum() else "_" for c in str(country).lower())
        cache_file = os.path.join(vec_dir, f"{os.path.basename(cache_key)}_{safe_country}.npy")
        if os.path.exists(cache_file):
            try:
                emb = np.load(cache_file).astype(np.float32)
                if len(emb) == len(df):
                    return emb
            except Exception:
                pass

    emb = model.encode(_combined_text(df), batch_size=batch_size,
                       show_progress_bar=True, normalize_embeddings=True,
                       convert_to_numpy=True).astype(np.float32)

    if cache_file:
        try:
            np.save(cache_file, emb.astype(np.float16))
        except Exception as e:
            print(f"    Warning: Could not save vector cache: {e}")

    return emb


def embedding_knn_candidates(s1_df, other_df, top_k=None, model_name=None, batch_size=None, cache_path=None, s1_tag=None, other_tag=None):
    """
    For each S1 record, find the top_k nearest Source-2/3 records by cosine
    similarity on sentence embeddings of (normalized name + address).
    Grouped by country first — a business in France won't be a neighbour of
    one in the US, so this also caps compute.

    When CFG.USE_FAISS_GPU is True (auto-detected), uses FAISS-GPU for
    sub-second kNN search over millions of vectors.
    """
    if cache_path and os.path.exists(cache_path):
        return load_candidates_cache(cache_path)

    import faiss
    from sentence_transformers import SentenceTransformer

    top_k = top_k or CFG.TOP_K_EMBEDDING
    model_name = model_name or CFG.EMBEDDING_MODEL
    batch_size = batch_size or CFG.EMBEDDING_BATCH_SIZE
    model = SentenceTransformer(model_name, device=CFG.DEVICE)
    if CFG.DEVICE == "cuda":
        model.half()  # FP16 — halves VRAM, doubles throughput on 80 GB GPU

    candidates = defaultdict(set)

    for country, s1_group in tqdm(s1_df.groupby("country"), desc="  Embedding kNN by country"):
        other_group = other_df[other_df["country"] == country]
        if len(other_group) == 0 or len(s1_group) == 0:
            continue

        s1_emb = _get_or_compute_embeddings(s1_group, country, model, batch_size, cache_key=s1_tag)
        
        # Inner-product on L2-normalized vectors = cosine similarity
        dim = s1_emb.shape[1]
        cpu_index = faiss.IndexFlatIP(dim)
        
        if CFG.USE_FAISS_GPU:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            index = cpu_index
            
        # CRITICAL: Chunk the encoding of other_group to prevent Kaggle OOM!
        # Encoding 5M strings at once takes 15GB of RAM and crashes the 16GB Kaggle kernel.
        chunk_size = 250_000
        for start_idx in tqdm(range(0, len(other_group), chunk_size), desc=f"    Encoding {len(other_group):,} targets"):
            chunk_df = other_group.iloc[start_idx : start_idx + chunk_size]
            # Use raw encode instead of cache wrapper so we can stream it directly into FAISS
            emb_chunk = model.encode(
                _combined_text(chunk_df), 
                batch_size=batch_size,
                show_progress_bar=False, 
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype(np.float32)
            index.add(emb_chunk)
            del emb_chunk
            gc.collect()

        k = min(top_k, len(other_group))
        _, idx = index.search(s1_emb, k)

        other_ids = other_group["entity_id"].values
        s1_ids = s1_group["entity_id"].values
        for i, s1_id in enumerate(s1_ids):
            candidates[s1_id].update(other_ids[j] for j in idx[i] if j >= 0)

        del index, cpu_index
        gc.collect()

    result = dict(candidates)
    if cache_path:
        save_candidates_cache(result, cache_path)

    return result


def generate_candidates(s1_df, s2_df, s3_df, use_embeddings=None, cache_prefix=None):
    """
    Union of string-key blocking and (optionally) embedding kNN, across both
    Source 2 and Source 3. Returns {s1_id: set(candidate_ids)}.
    Optionally caches intermediate and combined results to disk.
    """
    use_embeddings = CFG.USE_EMBEDDING_BLOCKING if use_embeddings is None else use_embeddings

    if cache_prefix:
        combined_cache = f"{cache_prefix}_combined.parquet"
        if os.path.exists(combined_cache):
            print(f"Loading cached combined candidates from {combined_cache}...")
            return load_candidates_cache(combined_cache)

    s2_cache = f"{cache_prefix}_s2.parquet" if cache_prefix else None
    s3_cache = f"{cache_prefix}_s3.parquet" if cache_prefix else None

    print("Generating string-key candidates (Source 2)...")
    key_s2 = string_key_candidates(s1_df, s2_df, cache_path=s2_cache)
    print("Generating string-key candidates (Source 3)...")
    key_s3 = string_key_candidates(s1_df, s3_df, cache_path=s3_cache)

    combined = defaultdict(set)
    for s1_id in s1_df["entity_id"]:
        if s1_id in key_s2:
            combined[s1_id].update(key_s2[s1_id])
        if s1_id in key_s3:
            combined[s1_id].update(key_s3[s1_id])

    del key_s2, key_s3
    gc.collect()

    if use_embeddings:
        emb_s2_cache = f"{cache_prefix}_emb_s2.parquet" if cache_prefix else None
        emb_s3_cache = f"{cache_prefix}_emb_s3.parquet" if cache_prefix else None
        s1_tag = f"{cache_prefix}_s1" if cache_prefix else None
        s2_tag = f"{cache_prefix}_s2" if cache_prefix else None
        s3_tag = f"{cache_prefix}_s3" if cache_prefix else None

        print("Generating embedding kNN candidates (Source 2)...")
        emb_s2 = embedding_knn_candidates(s1_df, s2_df, cache_path=emb_s2_cache, s1_tag=s1_tag, other_tag=s2_tag)
        print("Generating embedding kNN candidates (Source 3)...")
        emb_s3 = embedding_knn_candidates(s1_df, s3_df, cache_path=emb_s3_cache, s1_tag=s1_tag, other_tag=s3_tag)
        for s1_id in s1_df["entity_id"]:
            if s1_id in emb_s2:
                combined[s1_id].update(emb_s2[s1_id])
            if s1_id in emb_s3:
                combined[s1_id].update(emb_s3[s1_id])
        del emb_s2, emb_s3
        gc.collect()

    result = dict(combined)
    if cache_prefix:
        save_candidates_cache(result, f"{cache_prefix}_combined.parquet")

    return result


def candidate_recall(candidates: dict, ground_truth: dict) -> float:
    """
    Sanity check before you build the matcher: what fraction of true matches
    are even reachable from your candidate set? This is your recall ceiling
    — the matching model can never exceed it. Run this first.
    """
    hit, total = 0, 0
    for s1_id, true_matches in tqdm(ground_truth.items(), desc="Calculating candidate recall"):
        if s1_id not in candidates or not true_matches:
            continue  # entity not in this candidate set (e.g. a val-only subset) -> not this call's concern
        cand = candidates[s1_id]
        hit += len(true_matches & cand)
        total += len(true_matches)
    return hit / total if total else 1.0
