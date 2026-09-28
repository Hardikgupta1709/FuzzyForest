"""
run_train_full.py — Full-scale training on ALL training data.

Strategy to fit in 8GB RAM:
1. Load S1 entirely (200MB, fits easily)
2. Load ground truth (121MB, fits easily)
3. For each country:
   a. Load S2+S3 for that country only (chunked reading with country filter)
   b. Normalize
   c. Run blocking
   d. Build training pairs
   e. Compute features
   f. Free memory
4. Merge all pairs, train LightGBM

Memory estimates per country:
- India: ~4.7M pool records × ~100 bytes = ~470MB + working space = ~2GB
- US: ~3.8M pool records = ~380MB + working space = ~1.5GB
- Only US+India in training data, no France
"""
import sys, os, gc, random, csv, pickle
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

# === CONFIGURATION ===
SAMPLE_FRAC = None  # Set to e.g. 0.1 for 10% sample, None for full
MAX_NEG_PER_POS = 5

print("=" * 70)
print("FULL-SCALE TRAINING PIPELINE")
print("=" * 70)

# 1. Load ground truth (121MB, fits easily)
print("\n[1] Loading ground truth...")
gt = load_ground_truth(config.TRAIN_GT)
print(f"  {len(gt)} S1 entities in ground truth")

# 2. Create train/val split FIRST (entity-level)
train_ids, val_ids = create_train_val_split(gt)

# 3. Load S1 entirely (~200MB)
print("\n[2] Loading S1...")
s1_all = pd.read_csv(config.TRAIN_S1, sep="\t", dtype=str, keep_default_na=False)
countries = s1_all["country"].unique()
print(f"  {len(s1_all)} S1 records, countries: {list(countries)}")

if SAMPLE_FRAC:
    s1_all = s1_all.sample(frac=SAMPLE_FRAC, random_state=42)
    print(f"  Sampled to {len(s1_all)} records")

# 4. Process each country
all_X_train = []
all_y_train = []
all_X_val = []
all_y_val = []
all_val_pairs = []
total_blocking_recall_num = 0
total_blocking_recall_den = 0

for country in countries:
    print(f"\n{'='*50}")
    print(f"COUNTRY: {country}")
    print(f"{'='*50}")
    
    # Filter S1
    s1_df = s1_all[s1_all["country"] == country].copy()
    s1_entity_ids = set(s1_df["entity_id"])
    
    print(f"  S1: {len(s1_df)} entities")
    
    # Load S2+S3 for this country only (chunked to save memory)
    print(f"  Loading S2 for {country}...")
    s2_chunks = []
    for chunk in pd.read_csv(config.TRAIN_S2, sep="\t", dtype=str, 
                              keep_default_na=False, chunksize=500_000):
        s2_chunks.append(chunk[chunk["country"] == country])
    s2_df = pd.concat(s2_chunks, ignore_index=True) if s2_chunks else pd.DataFrame()
    del s2_chunks
    gc.collect()
    
    print(f"  Loading S3 for {country}...")
    s3_chunks = []
    for chunk in pd.read_csv(config.TRAIN_S3, sep="\t", dtype=str,
                              keep_default_na=False, chunksize=500_000):
        s3_chunks.append(chunk[chunk["country"] == country])
    s3_df = pd.concat(s3_chunks, ignore_index=True) if s3_chunks else pd.DataFrame()
    del s3_chunks
    gc.collect()
    
    pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    
    print(f"  Pool (S2+S3): {len(pool_df)} records")
    
    if SAMPLE_FRAC and len(pool_df) > 50000:
        pool_df = pool_df.sample(n=50000, random_state=42)
        print(f"  Sampled pool to {len(pool_df)} records")
    
    # Normalize
    print("  Normalizing...")
    s1_df = normalize_dataframe(s1_df)
    pool_df = normalize_dataframe(pool_df)
    
    # Blocking
    print("  Blocking...")
    gt_country = {k: v for k, v in gt.items() if k in s1_entity_ids}
    candidates = run_all_blocking(s1_df, pool_df, country=country, ground_truth=gt_country)
    
    # Ensure every S1 has entry
    for eid in s1_df["entity_id"]:
        if eid not in candidates:
            candidates[eid] = set()
    
    # Build data dicts
    s1_data = build_record_dict(s1_df)
    pool_data = build_record_dict(pool_df)
    
    # Split into train/val for this country
    train_s1_country = train_ids & s1_entity_ids
    val_s1_country = val_ids & s1_entity_ids
    
    # Build training pairs
    print("  Building training pairs...")
    train_pairs = build_training_pairs(train_s1_country, candidates, gt, neg_pos_ratio=MAX_NEG_PER_POS)
    val_pairs = build_training_pairs(val_s1_country, candidates, gt, neg_pos_ratio=MAX_NEG_PER_POS)
    
    # Compute features
    if train_pairs:
        print(f"  Computing features for {len(train_pairs)} train pairs...")
        X_t = np.zeros((len(train_pairs), config.NUM_FEATURES), dtype=np.float32)
        y_t = np.zeros(len(train_pairs), dtype=np.int32)
        for i, (s1_id, cand_id, label) in enumerate(tqdm(train_pairs, desc="Train features")):
            s1 = s1_data.get(s1_id, {})
            cand = pool_data.get(cand_id, {})
            X_t[i] = compute_pair_features(
                s1.get("norm_name", ""), cand.get("norm_name", ""),
                s1.get("norm_addr", ""), cand.get("norm_addr", ""),
                s1.get("raw_addr", ""), cand.get("raw_addr", ""),
                0.0,  # embedding_cosine (not available locally)
                cand_id.startswith("S3-"),
            )
            y_t[i] = label
        all_X_train.append(X_t)
        all_y_train.append(y_t)
    
    if val_pairs:
        print(f"  Computing features for {len(val_pairs)} val pairs...")
        X_v = np.zeros((len(val_pairs), config.NUM_FEATURES), dtype=np.float32)
        y_v = np.zeros(len(val_pairs), dtype=np.int32)
        for i, (s1_id, cand_id, label) in enumerate(tqdm(val_pairs, desc="Val features")):
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
    
    # Free everything
    del s1_df, pool_df, candidates, s1_data, pool_data, train_pairs, val_pairs
    gc.collect()

# 5. Merge all countries
print(f"\n{'='*50}")
print("MERGING AND TRAINING")
print(f"{'='*50}")

X_train = np.concatenate(all_X_train) if all_X_train else np.zeros((0, config.NUM_FEATURES))
y_train = np.concatenate(all_y_train) if all_y_train else np.zeros(0)
X_val = np.concatenate(all_X_val) if all_X_val else np.zeros((0, config.NUM_FEATURES))
y_val = np.concatenate(all_y_val) if all_y_val else np.zeros(0)

del all_X_train, all_y_train, all_X_val, all_y_val
gc.collect()

print(f"Total train: {X_train.shape} (pos={y_train.sum():.0f}, neg={len(y_train)-y_train.sum():.0f})")
print(f"Total val: {X_val.shape} (pos={y_val.sum():.0f}, neg={len(y_val)-y_val.sum():.0f})")

# 6. Train
print("\nTraining LightGBM...")
model = train_model(X_train, y_train, X_val, y_val)

# 7. Threshold calibration
print("\nCalibrating threshold...")
best_t, best_f05, results = find_optimal_threshold(model, X_val, all_val_pairs, gt)

# 8. Save
save_model(model)
threshold_path = os.path.join(config.MODEL_DIR, "threshold.txt")
os.makedirs(config.MODEL_DIR, exist_ok=True)
with open(threshold_path, "w") as f:
    f.write(str(best_t))

print(f"\n{'='*70}")
print(f"TRAINING COMPLETE")
print(f"  Best F0.5: {best_f05:.4f}")
print(f"  Threshold: {best_t:.2f}")
print(f"  Model: {config.MODEL_PATH}")
print(f"{'='*70}")
