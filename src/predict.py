"""
predict.py — End-to-end inference pipeline for entity resolution.

Processes each country partition separately to fit in 8GB RAM.
Generates both matching_results.tsv and candidate_pairs.tsv.
"""

import os
import csv
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

from . import config
from .normalize import normalize_name, normalize_address
from .features import compute_pair_features
from .train import load_model


def load_source_data(path: str) -> pd.DataFrame:
    """Load a source TSV file."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized name/address columns to a dataframe."""
    tqdm.pandas(desc="Normalizing names")
    df["norm_name"] = df["business_name"].progress_apply(normalize_name)
    tqdm.pandas(desc="Normalizing addresses")
    df["norm_addr"] = df["business_address"].progress_apply(normalize_address)
    return df


def build_record_dict(df: pd.DataFrame) -> dict:
    """Convert dataframe to lookup dict: {entity_id: {norm_name, norm_addr, raw_addr}}."""
    records = {}
    for _, row in df.iterrows():
        records[row["entity_id"]] = {
            "norm_name": row.get("norm_name", ""),
            "norm_addr": row.get("norm_addr", ""),
            "raw_addr": row.get("business_address", ""),
        }
    return records


def predict_country_partition(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    model: lgb.Booster,
    threshold: float,
    country: str,
    blocking_fn=None,
    embeddings_s1: dict = None,
    embeddings_pool: dict = None,
):
    """
    Run the full pipeline for one country partition.
    
    Returns:
        matching: {s1_id: set(matched_ids)}  
        candidates: {s1_id: set(candidate_ids)}
    """
    print(f"\n{'='*60}")
    print(f"Processing country: {country}")
    print(f"  S1 entities: {len(s1_df)}")
    print(f"  Pool (S2+S3): {len(pool_df)}")
    print(f"{'='*60}")
    
    # 1. Normalize
    print("Step 1: Normalizing...")
    s1_df = normalize_dataframe(s1_df.copy())
    pool_df = normalize_dataframe(pool_df.copy())
    
    # 2. Build record dicts
    s1_records = build_record_dict(s1_df)
    pool_records = build_record_dict(pool_df)
    
    # 3. Run blocking
    print("Step 2: Blocking (candidate generation)...")
    if blocking_fn is not None:
        candidates = blocking_fn(s1_df, pool_df)
    else:
        # Default: import and run all blocking
        from .blocking import run_all_blocking
        candidates = run_all_blocking(s1_df, pool_df, country=country)
    
    # Ensure every S1 entity has an entry
    for eid in s1_df["entity_id"]:
        if eid not in candidates:
            candidates[eid] = set()
    
    total_cands = sum(len(v) for v in candidates.values())
    non_empty = sum(1 for v in candidates.values() if v)
    print(f"  Total candidates: {total_cands}")
    print(f"  Non-empty S1 entities: {non_empty}/{len(candidates)}")
    print(f"  Avg candidates/entity: {total_cands/max(len(candidates),1):.1f}")
    
    # 4. Compute features and predict in batches
    print("Step 3: Feature computation + inference...")
    matching = {}
    
    batch_s1_ids = list(candidates.keys())
    
    for batch_start in tqdm(range(0, len(batch_s1_ids), 1000), desc="Inference batches"):
        batch_ids = batch_s1_ids[batch_start:batch_start + 1000]
        
        # Collect all pairs for this batch
        pairs = []
        for s1_id in batch_ids:
            for cand_id in candidates[s1_id]:
                pairs.append((s1_id, cand_id))
        
        if not pairs:
            for s1_id in batch_ids:
                matching[s1_id] = set()
            continue
        
        # Compute features
        X = np.zeros((len(pairs), config.NUM_FEATURES), dtype=np.float32)
        for i, (s1_id, cand_id) in enumerate(pairs):
            s1 = s1_records.get(s1_id, {})
            cand = pool_records.get(cand_id, {})
            
            emb_cos = 0.0
            if embeddings_s1 and embeddings_pool:
                emb_a = embeddings_s1.get(s1_id)
                emb_b = embeddings_pool.get(cand_id)
                if emb_a is not None and emb_b is not None:
                    dot = np.dot(emb_a, emb_b)
                    norm = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
                    emb_cos = float(dot / norm) if norm > 0 else 0.0
            
            X[i] = compute_pair_features(
                norm_name_a=s1.get("norm_name", ""),
                norm_name_b=cand.get("norm_name", ""),
                norm_addr_a=s1.get("norm_addr", ""),
                norm_addr_b=cand.get("norm_addr", ""),
                raw_addr_a=s1.get("raw_addr", ""),
                raw_addr_b=cand.get("raw_addr", ""),
                emb_cosine=emb_cos,
                source_is_s3=cand_id.startswith("S3-"),
            )
        
        # Predict
        probas = model.predict(X)
        
        # Apply threshold
        pair_idx = 0
        for s1_id in batch_ids:
            matched = set()
            n_cands = len(candidates[s1_id])
            for j in range(n_cands):
                if pair_idx < len(probas) and probas[pair_idx] >= threshold:
                    matched.add(pairs[pair_idx][1])
                pair_idx += 1
            matching[s1_id] = matched
        
        del X, probas
    
    return matching, candidates


def write_matching_results(matching: dict, all_s1_ids: list, output_path: str):
    """
    Write matching_results.tsv.
    
    Rules enforced:
    - One row per S1 entity (ALL test S1 entities)
    - Empty matched_entity_ids for singletons
    - No duplicate IDs within a list
    - Only S2-/S3- IDs
    - No duplicate source1_entity_id rows
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    seen = set()
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        
        for s1_id in all_s1_ids:
            assert s1_id not in seen, f"Duplicate S1 entity: {s1_id}"
            seen.add(s1_id)
            
            matched = matching.get(s1_id, set())
            # Filter: only S2-/S3- IDs, deduplicate
            matched = {mid for mid in matched if mid.startswith(("S2-", "S3-"))}
            
            matched_str = ",".join(sorted(matched)) if matched else ""
            writer.writerow([s1_id, matched_str])
    
    print(f"Written {len(seen)} rows to {output_path}")
    print(f"  Non-empty: {sum(1 for s1 in all_s1_ids if matching.get(s1))}")
    print(f"  Singletons: {sum(1 for s1 in all_s1_ids if not matching.get(s1))}")


def write_candidate_pairs(candidates: dict, all_s1_ids: list, output_path: str):
    """
    Write candidate_pairs.tsv.
    
    Same format rules as matching_results.tsv but with candidate_entity_ids column.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        
        for s1_id in all_s1_ids:
            cands = candidates.get(s1_id, set())
            cands = {cid for cid in cands if cid.startswith(("S2-", "S3-"))}
            cand_str = ",".join(sorted(cands)) if cands else ""
            writer.writerow([s1_id, cand_str])
    
    print(f"Written {len(all_s1_ids)} rows to {output_path}")


def run_full_prediction(
    model_path: str = None,
    threshold: float = 0.5,
    test_dir: str = None,
    output_dir: str = None,
    blocking_fn=None,
    embeddings_dir: str = None,
):
    """
    Run full prediction pipeline on test data.
    
    Processes each country partition separately for memory safety.
    """
    if model_path is None:
        model_path = config.MODEL_PATH
    if test_dir is None:
        test_dir = config.TEST_DIR
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    # Load model
    model = load_model(model_path)
    
    # Load test data
    print("Loading test data...")
    s1_df = load_source_data(os.path.join(test_dir, "test_source1.tsv"))
    s2_df = load_source_data(os.path.join(test_dir, "test_source2.tsv"))
    s3_df = load_source_data(os.path.join(test_dir, "test_source3.tsv"))
    
    # Combine S2+S3 pool
    pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    
    all_s1_ids = s1_df["entity_id"].tolist()
    countries = s1_df["country"].unique()
    print(f"Countries in test: {list(countries)}")
    print(f"Total S1 entities: {len(all_s1_ids)}")
    
    # Process per country
    all_matching = {}
    all_candidates = {}
    
    for country in countries:
        s1_country = s1_df[s1_df["country"] == country].copy()
        pool_country = pool_df[pool_df["country"] == country].copy()
        
        # Load embeddings for this country if available
        emb_s1 = None
        emb_pool = None
        if embeddings_dir:
            emb_path_s1 = os.path.join(embeddings_dir, f"embeddings_s1_{country}.npz")
            emb_path_pool = os.path.join(embeddings_dir, f"embeddings_pool_{country}.npz")
            if os.path.exists(emb_path_s1) and os.path.exists(emb_path_pool):
                print(f"Loading embeddings for {country}...")
                # Load as dict: {entity_id: embedding_vector}
                data_s1 = np.load(emb_path_s1, allow_pickle=True)
                emb_s1 = dict(zip(data_s1["ids"], data_s1["embeddings"]))
                data_pool = np.load(emb_path_pool, allow_pickle=True)
                emb_pool = dict(zip(data_pool["ids"], data_pool["embeddings"]))
        
        matching, candidates = predict_country_partition(
            s1_country, pool_country, model, threshold, country,
            blocking_fn=blocking_fn,
            embeddings_s1=emb_s1,
            embeddings_pool=emb_pool,
        )
        
        all_matching.update(matching)
        all_candidates.update(candidates)
        
        # Free memory
        del s1_country, pool_country, matching, candidates, emb_s1, emb_pool
        gc.collect()
    
    # Verify every S1 entity has an entry
    for s1_id in all_s1_ids:
        if s1_id not in all_matching:
            all_matching[s1_id] = set()
        if s1_id not in all_candidates:
            all_candidates[s1_id] = set()
    
    # Verify matches ⊆ candidates
    violations = 0
    for s1_id in all_s1_ids:
        extra = all_matching[s1_id] - all_candidates[s1_id]
        if extra:
            violations += 1
            # Fix by adding to candidates
            all_candidates[s1_id] |= extra
    if violations:
        print(f"WARNING: {violations} S1 entities had matches not in candidates (auto-fixed)")
    
    # Write outputs
    write_matching_results(
        all_matching, all_s1_ids,
        os.path.join(output_dir, "matching_results.tsv"),
    )
    write_candidate_pairs(
        all_candidates, all_s1_ids,
        os.path.join(output_dir, "candidate_pairs.tsv"),
    )
    
    print(f"\n✅ Prediction complete!")
    print(f"  matching_results.tsv: {len(all_s1_ids)} rows")
    print(f"  candidate_pairs.tsv: {len(all_s1_ids)} rows")
