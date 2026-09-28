"""
run_predict_test.py — Full-scale prediction on test data.

Strategy to fit in 8GB RAM:
- Process each country separately
- For each country: load pool, normalize, block, compute features, predict
- Merge all country results at the end
"""
import sys, os, gc
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "/Users/hardik/Documents/Amazon Ml Challenge/student_resource/code/business_entity_resolution")

from src import config
from src.normalize import normalize_name, normalize_address
from src.blocking import run_all_blocking
from src.features import compute_pair_features
from src.train import load_model
from src.predict import (
    load_source_data, normalize_dataframe, build_record_dict,
    write_matching_results, write_candidate_pairs,
)

# Load threshold
threshold_path = os.path.join(config.MODEL_DIR, "threshold.txt")
if os.path.exists(threshold_path):
    with open(threshold_path) as f:
        THRESHOLD = float(f.read().strip())
else:
    THRESHOLD = 0.50

print("=" * 70)
print("FULL-SCALE TEST PREDICTION")
print(f"  Threshold: {THRESHOLD:.2f}")
print("=" * 70)

# Load model
model = load_model()

# Load S1 (test) — ~167MB, fits easily
print("\n[1] Loading test S1...")
s1_all = pd.read_csv(config.TEST_S1, sep="\t", dtype=str, keep_default_na=False)
all_s1_ids = s1_all["entity_id"].tolist()
countries = s1_all["country"].unique()
print(f"  {len(s1_all)} S1 entities, countries: {list(countries)}")

all_matching = {}
all_candidates = {}

for country in countries:
    print(f"\n{'='*50}")
    print(f"COUNTRY: {country}")
    print(f"{'='*50}")
    
    s1_df = s1_all[s1_all["country"] == country].copy()
    s1_entity_ids = list(s1_df["entity_id"])
    print(f"  S1: {len(s1_df)} entities")
    
    # Load S2+S3 for this country (chunked)
    print(f"  Loading S2 for {country}...")
    s2_chunks = []
    for chunk in pd.read_csv(config.TEST_S2, sep="\t", dtype=str,
                              keep_default_na=False, chunksize=500_000):
        s2_chunks.append(chunk[chunk["country"] == country])
    s2_df = pd.concat(s2_chunks, ignore_index=True) if s2_chunks else pd.DataFrame()
    del s2_chunks
    gc.collect()
    
    print(f"  Loading S3 for {country}...")
    s3_chunks = []
    for chunk in pd.read_csv(config.TEST_S3, sep="\t", dtype=str,
                              keep_default_na=False, chunksize=500_000):
        s3_chunks.append(chunk[chunk["country"] == country])
    s3_df = pd.concat(s3_chunks, ignore_index=True) if s3_chunks else pd.DataFrame()
    del s3_chunks
    gc.collect()
    
    pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    
    print(f"  Pool (S2+S3): {len(pool_df)} records")
    
    # Normalize
    print("  Normalizing...")
    s1_df = normalize_dataframe(s1_df)
    pool_df = normalize_dataframe(pool_df)
    
    # Blocking
    print("  Blocking...")
    candidates = run_all_blocking(s1_df, pool_df, country=country)
    
    # Ensure every S1 has entry
    for eid in s1_entity_ids:
        if eid not in candidates:
            candidates[eid] = set()
    
    total_cands = sum(len(v) for v in candidates.values())
    print(f"  Total candidates: {total_cands}, avg per entity: {total_cands/max(len(s1_entity_ids),1):.1f}")
    
    # Build data dicts
    s1_data = build_record_dict(s1_df)
    pool_data = build_record_dict(pool_df)
    
    # Feature computation + inference in batches
    print("  Inference...")
    matching = {}
    
    for batch_start in tqdm(range(0, len(s1_entity_ids), 500), desc=f"Inference {country}"):
        batch_ids = s1_entity_ids[batch_start:batch_start + 500]
        
        # Collect all pairs for this batch
        pairs = []
        for s1_id in batch_ids:
            for cand_id in candidates.get(s1_id, set()):
                pairs.append((s1_id, cand_id))
        
        if not pairs:
            for s1_id in batch_ids:
                matching[s1_id] = set()
            continue
        
        # Compute features
        X = np.zeros((len(pairs), config.NUM_FEATURES), dtype=np.float32)
        for i, (s1_id, cand_id) in enumerate(pairs):
            s1 = s1_data.get(s1_id, {})
            cand = pool_data.get(cand_id, {})
            X[i] = compute_pair_features(
                s1.get("norm_name", ""), cand.get("norm_name", ""),
                s1.get("norm_addr", ""), cand.get("norm_addr", ""),
                s1.get("raw_addr", ""), cand.get("raw_addr", ""),
                0.0,  # embedding_cosine
                cand_id.startswith("S3-"),
            )
        
        # Predict
        probas = model.predict(X)
        
        # Apply threshold and group by S1
        pair_idx = 0
        for s1_id in batch_ids:
            matched = set()
            n_cands = len(candidates.get(s1_id, set()))
            for j in range(n_cands):
                if pair_idx < len(probas) and probas[pair_idx] >= THRESHOLD:
                    matched.add(pairs[pair_idx][1])
                pair_idx += 1
            matching[s1_id] = matched
        
        del X, probas
    
    all_matching.update(matching)
    for s1_id in s1_entity_ids:
        all_candidates[s1_id] = candidates.get(s1_id, set())
    
    del s1_df, pool_df, candidates, s1_data, pool_data, matching
    gc.collect()

# Ensure every S1 entity has an entry
for s1_id in all_s1_ids:
    if s1_id not in all_matching:
        all_matching[s1_id] = set()
    if s1_id not in all_candidates:
        all_candidates[s1_id] = set()

# Verify matches ⊆ candidates
for s1_id in all_s1_ids:
    extra = all_matching[s1_id] - all_candidates[s1_id]
    if extra:
        all_candidates[s1_id] |= extra

# Write outputs
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
write_matching_results(all_matching, all_s1_ids, config.MATCHING_RESULTS)
write_candidate_pairs(all_candidates, all_s1_ids, config.CANDIDATE_PAIRS)

print(f"\n{'='*70}")
print("PREDICTION COMPLETE")
print(f"  matching_results.tsv: {len(all_s1_ids)} rows")
print(f"  candidate_pairs.tsv: {len(all_s1_ids)} rows")
print(f"{'='*70}")
