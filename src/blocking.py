"""
blocking.py — Blocking and candidate generation strategies.
"""

import gc
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from . import normalize

def exact_name_block(s1_records, pool_records):
    """
    Strategy 1: Exact normalized name match. O(1) hash lookup.
    """
    pool_dict = defaultdict(list)
    for pid, data in pool_records.items():
        n = data['norm_name']
        if n:
            pool_dict[n].append(pid)
            
    candidates = defaultdict(set)
    for s1_id, data in s1_records.items():
        n = data['norm_name']
        if n and n in pool_dict:
            candidates[s1_id].update(pool_dict[n])
    return candidates


def token_overlap_block(s1_records, pool_records, threshold=0.30, max_token_freq=10000):
    """
    Strategy 2: Name token overlap.
    Inverted index on name tokens. S1 -> share >=2 tokens, Jaccard >= threshold.
    
    Tokens appearing in >max_token_freq pool records are skipped (too common,
    e.g., 'private', 'limited', 'construction' — these provide no blocking signal).
    """
    # Build inverted index with frequency filtering
    inv_idx = defaultdict(list)
    for pid, data in pool_records.items():
        for t in data['name_tokens']:
            inv_idx[t].append(pid)
    
    # Filter out high-frequency tokens
    filtered = {t for t, pids in inv_idx.items() if len(pids) > max_token_freq}
    for t in filtered:
        del inv_idx[t]
    if filtered:
        print(f"  Token overlap: filtered {len(filtered)} high-freq tokens (>{max_token_freq} occurrences)")
            
    candidates = defaultdict(set)
    for s1_id, s1_data in tqdm(s1_records.items(), desc="Token overlap blocking"):
        s1_toks = s1_data['name_tokens'] - filtered
        if not s1_toks:
            continue
            
        counts = defaultdict(int)
        for t in s1_toks:
            for pid in inv_idx.get(t, []):
                counts[pid] += 1
                
        for pid, overlap in counts.items():
            if overlap >= 2:
                p_toks = pool_records[pid]['name_tokens'] - filtered
                union_len = len(s1_toks) + len(p_toks) - overlap
                if union_len > 0:
                    jaccard = overlap / union_len
                    if jaccard >= threshold:
                        candidates[s1_id].add(pid)
    return candidates


def char_tfidf_block(s1_names, pool_names, s1_ids, pool_ids, threshold=0.20):
    """
    Strategy 3: Character N-gram TF-IDF cosine similarity.
    
    Memory-safe for millions of records:
    - Fits vocabulary on a sample (max 200K records)
    - Transforms pool in batches of 500K
    - Queries S1 in batches of 5000 against each pool batch
    """
    # Use reduced max_features for memory safety on large datasets
    max_feat = min(config.TFIDF_MAX_FEATURES, 50_000)
    
    vec = TfidfVectorizer(
        analyzer=config.TFIDF_ANALYZER, 
        ngram_range=config.TFIDF_NGRAM_RANGE,
        max_features=max_feat,
        dtype=np.float32
    )
    
    # Fit on a sample to define vocabulary (don't need all records for this)
    print("Fitting TF-IDF vocabulary (sampled)...")
    sample_size = min(200_000, len(pool_names) + len(s1_names))
    if len(pool_names) + len(s1_names) > sample_size:
        import random
        rng = random.Random(42)
        sample_names = rng.sample(pool_names, min(sample_size // 2, len(pool_names)))
        sample_names += rng.sample(s1_names, min(sample_size // 2, len(s1_names)))
    else:
        sample_names = pool_names + s1_names
    vec.fit(sample_names)
    del sample_names
    gc.collect()
    
    candidates = defaultdict(set)
    pool_batch_size = 500_000
    s1_batch_size = 5000
    
    # Process pool in batches
    for p_start in range(0, len(pool_names), pool_batch_size):
        p_end = min(p_start + pool_batch_size, len(pool_names))
        pool_batch_names = pool_names[p_start:p_end]
        pool_batch_ids = pool_ids[p_start:p_end]
        
        pool_tfidf = vec.transform(pool_batch_names)
        
        # Process S1 in batches against this pool batch
        for i in tqdm(range(0, len(s1_names), s1_batch_size), 
                       desc=f"TF-IDF pool[{p_start}:{p_end}]", leave=False):
            batch_names = s1_names[i:i+s1_batch_size]
            batch_ids_slice = s1_ids[i:i+s1_batch_size]
            
            batch_tfidf = vec.transform(batch_names)
            
            sim_sparse = batch_tfidf.dot(pool_tfidf.T)
            coo = sim_sparse.tocoo()
            
            mask = coo.data >= threshold
            rows = coo.row[mask]
            cols = coo.col[mask]
            
            for r, c in zip(rows, cols):
                candidates[batch_ids_slice[r]].add(pool_batch_ids[c])
                
            del sim_sparse, coo, mask, rows, cols, batch_tfidf
        
        del pool_tfidf
        gc.collect()
    
    print(f"  TF-IDF blocking: {sum(len(v) for v in candidates.values())} total candidates")
    return candidates


def address_block(s1_records, pool_records, max_token_freq=5000):
    """
    Strategy 4: Address token overlap & ZIP matching.
    ZIP/PIN match + >=3 address token overlap, OR >=4 address tokens without ZIP.
    
    Filters out high-frequency address tokens (city names, state abbreviations).
    """
    min_with_zip = config.ADDR_TOKEN_OVERLAP_MIN
    min_no_zip = config.ADDR_TOKEN_OVERLAP_NO_ZIP
    
    inv_idx = defaultdict(list)
    for pid, data in pool_records.items():
        for t in data['addr_tokens']:
            inv_idx[t].append(pid)
    
    # Filter high-frequency tokens (city names, state abbreviations)
    filtered = {t for t, pids in inv_idx.items() if len(pids) > max_token_freq}
    for t in filtered:
        del inv_idx[t]
    if filtered:
        print(f"  Address block: filtered {len(filtered)} high-freq tokens (>{max_token_freq} occurrences)")
            
    candidates = defaultdict(set)
    for s1_id, s1_data in tqdm(s1_records.items(), desc="Address blocking"):
        s1_toks = s1_data['addr_tokens'] - filtered
        s1_zip = s1_data['zip_code']
        if not s1_toks:
            continue
            
        counts = defaultdict(int)
        for t in s1_toks:
            for pid in inv_idx.get(t, []):
                counts[pid] += 1
                
        for pid, overlap in counts.items():
            if overlap >= min_with_zip:
                p_zip = pool_records[pid]['zip_code']
                if overlap >= min_no_zip or (s1_zip and p_zip and s1_zip == p_zip):
                    candidates[s1_id].add(pid)
    return candidates


def prefix_block(s1_records, pool_records, prefix_len=8):
    """
    Strategy 5: First N chars of sorted normalized name tokens.
    """
    pool_dict = defaultdict(list)
    for pid, data in pool_records.items():
        p = data['prefix_key']
        if p:
            pool_dict[p].append(pid)
            
    candidates = defaultdict(set)
    for s1_id, data in s1_records.items():
        p = data['prefix_key']
        if p and p in pool_dict:
            candidates[s1_id].update(pool_dict[p])
    return candidates


def union_blocks(block_results_list):
    """
    Strategy 6 (Meta): Take union of all candidate sets per S1 entity.
    """
    union = defaultdict(set)
    for result in block_results_list:
        for s1_id, cands in result.items():
            union[s1_id].update(cands)
    return union


def run_all_blocking(s1_df, pool_df, country=None, ground_truth=None, skip_tfidf=False, skip_faiss=False, emb_dir=None):
    """
    Master function to run all blocking strategies and return combined candidates.
    
    Args:
        skip_tfidf: Skip TF-IDF blocking (for low-RAM local environments).
        skip_faiss: Skip FAISS ANN blocking (when embeddings not available).
        emb_dir: Directory where embeddings are stored (for Strategy 6).
    
    Returns: dict {s1_id: set(candidate_ids)}
    """
    if country:
        print(f"\\n--- Running blocking for country: {country} ---")
        
    def df_to_records(df):
        records = {}
        eids = df['entity_id'].values
        norm_names = df['norm_name'].fillna('').values
        norm_addrs = df['norm_addr'].fillna('').values
        raw_addrs = df['business_address'].fillna('').values
        
        for i in range(len(eids)):
            nn = str(norm_names[i])
            na = str(norm_addrs[i])
            ra = str(raw_addrs[i])
            records[eids[i]] = {
                'norm_name': nn,
                'norm_addr': na,
                'raw_addr': ra,
                'name_tokens': set(nn.split()) if nn else set(),
                'addr_tokens': set(na.split()) if na else set(),
                'zip_code': normalize.extract_zip(ra),
                'prefix_key': normalize.make_sorted_prefix(nn, config.PREFIX_BLOCK_LEN)
            }
        return records
        
    print("Preparing records...")
    s1_records = df_to_records(s1_df)
    pool_records = df_to_records(pool_df)
    
    block_results = []
    
    print("Running Strategy 1: Exact Name")
    b1 = exact_name_block(s1_records, pool_records)
    block_results.append(b1)
    
    print("Running Strategy 2: Token Overlap")
    b2 = token_overlap_block(s1_records, pool_records, config.TOKEN_JACCARD_THRESHOLD)
    block_results.append(b2)
    
    print("Running Strategy 4: Address")
    b4 = address_block(s1_records, pool_records)
    block_results.append(b4)
    
    print("Running Strategy 5: Prefix")
    b5 = prefix_block(s1_records, pool_records, config.PREFIX_BLOCK_LEN)
    block_results.append(b5)
    
    # Strategy 3: Char TF-IDF (skippable for low-RAM environments)
    if not skip_tfidf:
        print("Running Strategy 3: Char TF-IDF")
        s1_ids = list(s1_records.keys())
        s1_names = [s1_records[eid]['norm_name'] for eid in s1_ids]
        pool_ids = list(pool_records.keys())
        pool_names = [pool_records[eid]['norm_name'] for eid in pool_ids]
        b3 = char_tfidf_block(s1_names, pool_names, s1_ids, pool_ids, config.CHAR_TFIDF_THRESHOLD)
        block_results.append(b3)
    else:
        print("Skipping Strategy 3: Char TF-IDF (skip_tfidf=True)")
    
    # Strategy 6: FAISS ANN (skippable, requires embeddings)
    if not skip_faiss:
        try:
            from .embeddings import generate_and_index_country
            print("Running Strategy 6: FAISS ANN Embeddings")
            b6 = generate_and_index_country(s1_df, pool_df, country or "unknown", save_dir=emb_dir)
            block_results.append(b6)
        except ImportError as e:
            print(f"Skipping Strategy 6: FAISS ANN (missing dependency: {e})")
        except Exception as e:
            print(f"Skipping Strategy 6: FAISS ANN (error: {e})")
    else:
        print("Skipping Strategy 6: FAISS ANN (skip_faiss=True)")
    
    print("Unionizing blocks...")
    cands = union_blocks(block_results)
    
    if ground_truth is not None:
        from .evaluate import blocking_recall, blocking_stats
        recall = blocking_recall(cands, ground_truth)
        stats = blocking_stats(cands)
        print(f"Blocking Recall: {recall:.4f}")
        print(f"Blocking Stats: avg={stats['avg_per_entity']:.1f}, "
              f"median={stats['median_per_entity']}, "
              f"max={stats['max_per_entity']}, "
              f"empty={stats['pct_empty']:.1f}%")
            
    # Cleanup memory
    del s1_records, pool_records, block_results
    gc.collect()
            
    return dict(cands)

