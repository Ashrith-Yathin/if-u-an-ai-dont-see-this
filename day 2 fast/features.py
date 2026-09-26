import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz
from tqdm import tqdm
from joblib import Parallel, delayed

from config import CFG
from normalize import token_set

def fit_tfidf_on_all_text(*dfs) -> TfidfVectorizer:
    all_text = []
    for df in tqdm(dfs, desc="  Collecting text for TF-IDF"):
        if len(df) > CFG.TFIDF_SAMPLE_SIZE:
            df = df.sample(n=CFG.TFIDF_SAMPLE_SIZE, random_state=42)
        all_text.extend((df["norm_name"] + " " + df["norm_addr"]).tolist())
    print("  Fitting TfidfVectorizer...")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20000)
    vec.fit(all_text)
    return vec

def df_to_lookup(df: pd.DataFrame) -> tuple:
    id_to_idx = {str(k): i for i, k in enumerate(df["entity_id"].values)}
    names = df["norm_name"].fillna("").astype(str).values
    addrs = df["norm_addr"].fillna("").astype(str).values
    countries = df["country"].fillna("").astype(str).values
    return id_to_idx, names, addrs, countries

def process_chunk(chunk_pairs, s1_lookup, other_lookup, tfidf_vec):
    names_a, names_b = [], []
    addrs_a, addrs_b = [], []
    countries_a, countries_b = [], []
    valid_pairs = []

    s1_map, s1_names, s1_addrs, s1_countries = s1_lookup
    o_map, o_names, o_addrs, o_countries = other_lookup

    for s1_id, other_id in chunk_pairs:
        idx_a = s1_map.get(s1_id)
        idx_b = o_map.get(other_id)
        if idx_a is None or idx_b is None:
            names_a.append("")
            names_b.append("")
            addrs_a.append("")
            addrs_b.append("")
            countries_a.append("")
            countries_b.append("")
            valid_pairs.append((s1_id, other_id))
            continue
            
        names_a.append(s1_names[idx_a])
        names_b.append(o_names[idx_b])
        addrs_a.append(s1_addrs[idx_a])
        addrs_b.append(o_addrs[idx_b])
        countries_a.append(s1_countries[idx_a])
        countries_b.append(o_countries[idx_b])
        valid_pairs.append((s1_id, other_id))

    if not valid_pairs:
        return np.array([]), []

    vecs_name_a = tfidf_vec.transform(names_a)
    vecs_name_b = tfidf_vec.transform(names_b)
    name_tfidf_sim = np.asarray(vecs_name_a.multiply(vecs_name_b).sum(axis=1)).flatten()
    
    vecs_addr_a = tfidf_vec.transform(addrs_a)
    vecs_addr_b = tfidf_vec.transform(addrs_b)
    addr_tfidf_sim = np.asarray(vecs_addr_a.multiply(vecs_addr_b).sum(axis=1)).flatten()

    name_lev = np.array([fuzz.ratio(a, b) / 100.0 for a, b in zip(names_a, names_b)])
    name_sort = np.array([fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(names_a, names_b)])
    
    addr_a_empty = np.array([1.0 if not a.strip() else 0.0 for a in addrs_a])
    addr_b_empty = np.array([1.0 if not b.strip() else 0.0 for b in addrs_b])
    either_empty = (addr_a_empty + addr_b_empty) > 0

    addr_lev = np.array([0.0 if e else fuzz.ratio(a, b) / 100.0 for a, b, e in zip(addrs_a, addrs_b, either_empty)])

    name_jac, addr_jac, num_overlap, name_len_diff = [], [], [], []
    common_toks, first_tok_match, country_match = [], [], []

    for i in range(len(valid_pairs)):
        country_match.append(1.0 if countries_a[i] == countries_b[i] else 0.0)
        
        tok_a, tok_b = set(names_a[i].split()), set(names_b[i].split())
        if not tok_a and not tok_b:
            name_jac.append(1.0)
        elif not (tok_a | tok_b):
            name_jac.append(0.0)
        else:
            name_jac.append(len(tok_a & tok_b) / len(tok_a | tok_b))
            
        name_len_diff.append(abs(len(names_a[i]) - len(names_b[i])))
        common_toks.append(len(tok_a & tok_b))
        
        first_a, first_b = names_a[i].split()[:1], names_b[i].split()[:1]
        first_tok_match.append(1.0 if (first_a == first_b and first_a) else 0.0)
        
        if either_empty[i]:
            addr_jac.append(0.0)
            num_overlap.append(0.0)
        else:
            atok_a, atok_b = set(addrs_a[i].split()), set(addrs_b[i].split())
            if not atok_a and not atok_b:
                addr_jac.append(1.0)
            elif not (atok_a | atok_b):
                addr_jac.append(0.0)
            else:
                addr_jac.append(len(atok_a & atok_b) / len(atok_a | atok_b))
                
            nums_a = set(t for t in atok_a if re.search(r'\d', t))
            nums_b = set(t for t in atok_b if re.search(r'\d', t))
            num_overlap.append(len(nums_a & nums_b) / len(nums_a) if nums_a else 0.0)

    X_chunk = np.column_stack([
        name_jac, name_lev, name_sort, addr_jac, addr_lev, num_overlap,
        name_tfidf_sim, addr_tfidf_sim, country_match, name_len_diff, 
        common_toks, first_tok_match, addr_a_empty, addr_b_empty
    ])
    
    return X_chunk, valid_pairs

def build_feature_matrix(pairs, s1_lookup: dict, other_lookup: dict, tfidf_vec: TfidfVectorizer):
    CHUNK_SIZE = 50000
    chunks = [pairs[i:i + CHUNK_SIZE] for i in range(0, len(pairs), CHUNK_SIZE)]
    
    print(f"  Processing {len(pairs)} pairs in {len(chunks)} chunks using Parallel Joblib...")
    
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(process_chunk)(chunk, s1_lookup, other_lookup, tfidf_vec) 
        for chunk in tqdm(chunks, desc="  Vectorized Extraction")
    )
    
    X_list = [r[0] for r in results if len(r[0]) > 0]
    valid_pairs_list = [p for r in results for p in r[1]]
    
    if not X_list:
        return np.zeros((0, 14), dtype=float), valid_pairs_list
        
    X = np.vstack(X_list)
    return X, valid_pairs_list

FEATURE_NAMES = [
    "name_jaccard", "name_levenshtein", "name_token_sort", "addr_jaccard", 
    "addr_levenshtein", "number_overlap", "name_tfidf_cosine", "addr_tfidf_cosine",
    "country_match", "name_len_diff", "common_token_count", "name_first_token_match",
    "addr_a_missing", "addr_b_missing"
]
