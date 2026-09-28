"""
run_fast_train.py — Fast training on a representative sample.

Strategy: Take 20% of S1 entities (stratified by country), include ALL their 
matching pool records plus random negatives. This gives us a high-quality model 
fast, which we use for the initial submission.
"""
import sys, os, gc, random, csv, time
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "/Users/hardik/Documents/Amazon Ml Challenge/student_resource/code/business_entity_resolution")

from src import config
from src.normalize import normalize_name, normalize_address
from src.evaluate import load_ground_truth, blocking_recall, blocking_stats
from src.blocking import run_all_blocking
from src.train import (
    create_train_val_split,
    build_training_pairs,
    train_model,
    find_optimal_threshold,
    save_model,
)
from src.features import compute_pair_features
from src.predict import normalize_dataframe, build_record_dict

start_time = time.time()
SAMPLE_FRAC = 0.05  # Reverted back to 5% as 10% caused OOM crash
POOL_RANDOM_PER_COUNTRY = 10000  # Reverted to 10000

print("=" * 70)
print(f"FAST TRAINING (sample={SAMPLE_FRAC*100:.0f}%)")
print("=" * 70)

# 1. Load ground truth
print("\n[1] Loading ground truth...")
gt = load_ground_truth(config.TRAIN_GT)
print(f"  {len(gt)} S1 entities")

# 2. Create train/val split on ALL entities first
all_train_ids, all_val_ids = create_train_val_split(gt)

# 3. Load S1
print("\n[2] Loading S1...")
s1_all = pd.read_csv(config.TRAIN_S1, sep="\t", dtype=str, keep_default_na=False)
countries = s1_all["country"].unique()
print(f"  {len(s1_all)} S1, countries: {list(countries)}")

all_X_train = []
all_y_train = []
all_X_val = []
all_y_val = []
all_val_pairs = []

for country in countries:
    print(f"\n{'='*50}")
    print(f"COUNTRY: {country}")
    print(f"{'='*50}")
    
    # Sample S1 for this country
    s1_country = s1_all[s1_all["country"] == country].copy()
    s1_sampled = s1_country.sample(frac=SAMPLE_FRAC, random_state=42)
    sample_s1_ids = set(s1_sampled["entity_id"])
    
    print(f"  S1 total: {len(s1_country)}, sampled: {len(s1_sampled)}")
    
    # Find all needed pool IDs for these S1 entities
    needed_pool = set()
    for sid in sample_s1_ids:
        needed_pool |= gt.get(sid, set())
    print(f"  Need {len(needed_pool)} matching pool records")
    
    # Load S2 for this country: needed + random
    print(f"  Loading S2...")
    s2_needed = []
    s2_random = []
    for chunk in pd.read_csv(config.TRAIN_S2, sep="\t", dtype=str,
                              keep_default_na=False, chunksize=500_000):
        chunk_country = chunk[chunk["country"] == country]
        s2_needed.append(chunk_country[chunk_country["entity_id"].isin(needed_pool)])
        s2_random.append(chunk_country[~chunk_country["entity_id"].isin(needed_pool)])
    
    s2_needed_df = pd.concat(s2_needed, ignore_index=True)
    s2_random_df = pd.concat(s2_random, ignore_index=True)
    if len(s2_random_df) > POOL_RANDOM_PER_COUNTRY:
        s2_random_df = s2_random_df.sample(n=POOL_RANDOM_PER_COUNTRY, random_state=42)
    s2_df = pd.concat([s2_needed_df, s2_random_df], ignore_index=True).drop_duplicates(subset=["entity_id"])
    del s2_needed, s2_random, s2_needed_df, s2_random_df
    gc.collect()
    
    # Load S3 same way
    print(f"  Loading S3...")
    s3_needed = []
    s3_random = []
    for chunk in pd.read_csv(config.TRAIN_S3, sep="\t", dtype=str,
                              keep_default_na=False, chunksize=500_000):
        chunk_country = chunk[chunk["country"] == country]
        s3_needed.append(chunk_country[chunk_country["entity_id"].isin(needed_pool)])
        s3_random.append(chunk_country[~chunk_country["entity_id"].isin(needed_pool)])
    
    s3_needed_df = pd.concat(s3_needed, ignore_index=True)
    s3_random_df = pd.concat(s3_random, ignore_index=True)
    if len(s3_random_df) > POOL_RANDOM_PER_COUNTRY:
        s3_random_df = s3_random_df.sample(n=POOL_RANDOM_PER_COUNTRY, random_state=42)
    s3_df = pd.concat([s3_needed_df, s3_random_df], ignore_index=True).drop_duplicates(subset=["entity_id"])
    del s3_needed, s3_random, s3_needed_df, s3_random_df
    gc.collect()
    
    pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    
    pool_ids_set = set(pool_df["entity_id"])
    covered = sum(1 for sid in sample_s1_ids if gt.get(sid, set()).issubset(pool_ids_set))
    print(f"  Pool: {len(pool_df)} records, {covered}/{len(sample_s1_ids)} S1 fully covered")
    
    # Normalize
    print("  Normalizing...")
    s1_sampled = normalize_dataframe(s1_sampled)
    pool_df = normalize_dataframe(pool_df)
    
    # Blocking
    print("  Blocking...")
    gt_sample = {k: v for k, v in gt.items() if k in sample_s1_ids}
    candidates = run_all_blocking(s1_sampled, pool_df, country=country, ground_truth=gt_sample,
                                    skip_tfidf=True, skip_faiss=True)
    
    for eid in s1_sampled["entity_id"]:
        if eid not in candidates:
            candidates[eid] = set()
    
    # Build data dicts
    s1_data = build_record_dict(s1_sampled)
    pool_data = build_record_dict(pool_df)
    
    # Split into train/val
    train_s1 = all_train_ids & sample_s1_ids
    val_s1 = all_val_ids & sample_s1_ids
    
    # Build pairs
    print("  Building pairs...")
    train_pairs = build_training_pairs(train_s1, candidates, gt, neg_pos_ratio=5)
    val_pairs = build_training_pairs(val_s1, candidates, gt, neg_pos_ratio=5)
    
    # Compute features for train pairs
    if train_pairs:
        print(f"  Computing train features ({len(train_pairs)} pairs)...")
        X_t = np.zeros((len(train_pairs), config.NUM_FEATURES), dtype=np.float32)
        y_t = np.zeros(len(train_pairs), dtype=np.int32)
        for i, (s1_id, cand_id, label) in enumerate(tqdm(train_pairs, desc=f"Train {country}")):
            s1 = s1_data.get(s1_id, {})
            cand = pool_data.get(cand_id, {})
            X_t[i] = compute_pair_features(
                s1.get("norm_name", ""), cand.get("norm_name", ""),
                s1.get("norm_addr", ""), cand.get("norm_addr", ""),
                s1.get("raw_addr", ""), cand.get("raw_addr", ""),
                0.0, cand_id.startswith("S3-"),
            )
            y_t[i] = label
        all_X_train.append(X_t)
        all_y_train.append(y_t)
    
    if val_pairs:
        print(f"  Computing val features ({len(val_pairs)} pairs)...")
        X_v = np.zeros((len(val_pairs), config.NUM_FEATURES), dtype=np.float32)
        y_v = np.zeros(len(val_pairs), dtype=np.int32)
        for i, (s1_id, cand_id, label) in enumerate(tqdm(val_pairs, desc=f"Val {country}")):
            s1 = s1_data.get(s1_id, {})
            cand = pool_data.get(cand_id, {})
            X_v[i] = compute_pair_features(
                s1.get("norm_name", ""), cand.get("norm_name", ""),
                s1.get("norm_addr", ""), cand.get("norm_addr", ""),
                s1.get("raw_addr", ""), cand.get("raw_addr", ""),
                0.0, cand_id.startswith("S3-"),
            )
            y_v[i] = label
        all_X_val.append(X_v)
        all_y_val.append(y_v)
        all_val_pairs.extend(val_pairs)
    
    del s1_sampled, pool_df, candidates, s1_data, pool_data, train_pairs, val_pairs
    gc.collect()
    elapsed = (time.time() - start_time) / 60
    print(f"  Elapsed: {elapsed:.1f} min")

# Merge
print(f"\n{'='*50}")
print("MERGING AND TRAINING")
print(f"{'='*50}")

X_train = np.concatenate(all_X_train)
y_train = np.concatenate(all_y_train)
X_val = np.concatenate(all_X_val)
y_val = np.concatenate(all_y_val)

del all_X_train, all_y_train, all_X_val, all_y_val
gc.collect()

print(f"Train: {X_train.shape} (pos={y_train.sum():.0f}, neg={len(y_train)-y_train.sum():.0f})")
print(f"Val: {X_val.shape} (pos={y_val.sum():.0f}, neg={len(y_val)-y_val.sum():.0f})")

# Train
model = train_model(X_train, y_train, X_val, y_val)

# Threshold
best_t, best_f05, results = find_optimal_threshold(model, X_val, all_val_pairs, gt)

# Save
save_model(model)
os.makedirs(config.MODEL_DIR, exist_ok=True)
with open(os.path.join(config.MODEL_DIR, "threshold.txt"), "w") as f:
    f.write(str(best_t))

elapsed = (time.time() - start_time) / 60
print(f"\n{'='*70}")
print(f"TRAINING COMPLETE in {elapsed:.1f} min")
print(f"  F0.5 = {best_f05:.4f}")
print(f"  Threshold = {best_t:.2f}")
print(f"{'='*70}")
