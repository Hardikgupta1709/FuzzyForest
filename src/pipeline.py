"""
pipeline.py — Master orchestrator for the entity resolution pipeline.

Usage:
    python -m src.pipeline --mode train       # Train on training data
    python -m src.pipeline --mode predict     # Run prediction on test data
    python -m src.pipeline --mode eval        # Evaluate predictions against ground truth
    python -m src.pipeline --mode sample      # Quick test on small sample
"""

import os
import sys
import gc
import argparse
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .normalize import normalize_name, normalize_address
from .evaluate import load_ground_truth, macro_f05, blocking_recall, blocking_stats
from .blocking import run_all_blocking
from .train import (
    create_train_val_split,
    build_training_pairs,
    compute_features_for_pairs,
    train_model,
    find_optimal_threshold,
    save_model,
    load_model,
)
from .predict import (
    load_source_data,
    normalize_dataframe,
    build_record_dict,
    write_matching_results,
    write_candidate_pairs,
    run_full_prediction,
)


def run_sample_test(n_sample: int = 5000):
    """
    Quick test on a small sample to verify the pipeline works end-to-end.
    """
    print("=" * 60)
    print(f"SAMPLE TEST (n={n_sample})")
    print("=" * 60)

    # Load small sample
    print("\n1. Loading data...")
    s1_df = pd.read_csv(config.TRAIN_S1, sep="\t", dtype=str, nrows=n_sample, keep_default_na=False)
    s2_df = pd.read_csv(config.TRAIN_S2, sep="\t", dtype=str, nrows=n_sample * 2, keep_default_na=False)
    s3_df = pd.read_csv(config.TRAIN_S3, sep="\t", dtype=str, nrows=n_sample * 2, keep_default_na=False)
    gt = load_ground_truth(config.TRAIN_GT)

    pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df

    # Filter GT to only include sampled S1 entities
    s1_ids = set(s1_df["entity_id"].tolist())
    gt_sample = {k: v for k, v in gt.items() if k in s1_ids}

    # Pick one country
    country = s1_df["country"].value_counts().idxmax()
    s1_country = s1_df[s1_df["country"] == country].head(min(n_sample, 2000))
    pool_country = pool_df[pool_df["country"] == country]

    print(f"  Country: {country}")
    print(f"  S1: {len(s1_country)}, Pool: {len(pool_country)}")

    # Normalize
    print("\n2. Normalizing...")
    s1_country = normalize_dataframe(s1_country.copy())
    pool_country = normalize_dataframe(pool_country.copy())

    # Blocking
    print("\n3. Blocking...")
    gt_country = {k: v for k, v in gt_sample.items() if k in set(s1_country["entity_id"])}
    candidates = run_all_blocking(s1_country, pool_country, country=country, ground_truth=gt_country)

    # Ensure every S1 has entry
    for eid in s1_country["entity_id"]:
        if eid not in candidates:
            candidates[eid] = set()

    total_cands = sum(len(v) for v in candidates.values())
    print(f"  Total candidates: {total_cands}")
    print(f"  Avg per entity: {total_cands / max(len(candidates), 1):.1f}")

    # Build training pairs
    print("\n4. Building training pairs...")
    s1_data = build_record_dict(s1_country)
    pool_data = build_record_dict(pool_country)

    train_ids, val_ids = create_train_val_split(gt_country)
    pairs = build_training_pairs(train_ids, candidates, gt_country, neg_pos_ratio=5)

    if not pairs:
        print("WARNING: No training pairs generated. Sample may be too small.")
        return

    # Compute features
    print("\n5. Computing features...")
    X, y = compute_features_for_pairs(pairs, s1_data, pool_data)
    print(f"  Feature matrix: {X.shape}, Labels: {y.shape}")
    print(f"  Positive: {(y == 1).sum()}, Negative: {(y == 0).sum()}")

    # Train
    print("\n6. Training LightGBM...")
    # Split into train/val by indices
    val_mask = np.array([p[0] in val_ids for p in pairs])
    train_mask = ~val_mask

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        print("WARNING: Not enough data for train/val split. Using all as train.")
        X_train, y_train = X, y
        X_val, y_val = X[:100], y[:100]
        val_pairs = pairs[:100]
    else:
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        val_pairs = [p for i, p in enumerate(pairs) if val_mask[i]]

    model = train_model(X_train, y_train, X_val, y_val)

    # Threshold calibration
    print("\n7. Calibrating threshold...")
    best_t, best_f05, _ = find_optimal_threshold(model, X_val, val_pairs, gt_country)

    print(f"\n{'=' * 60}")
    print(f"SAMPLE TEST COMPLETE")
    print(f"  Best threshold: {best_t:.2f}")
    print(f"  Best F0.5:      {best_f05:.4f}")
    print(f"{'=' * 60}")

    # Save model
    save_model(model)
    
    # Save threshold
    threshold_path = os.path.join(config.MODEL_DIR, "threshold.txt")
    os.makedirs(os.path.dirname(threshold_path), exist_ok=True)
    with open(threshold_path, "w") as f:
        f.write(str(best_t))
    print(f"Threshold saved to {threshold_path}")

    return model, best_t


def run_train(sample_frac: float = None):
    """
    Full training pipeline on training data.
    Processes per-country for memory safety.
    """
    print("=" * 60)
    print("FULL TRAINING PIPELINE")
    print("=" * 60)

    # Load ground truth
    gt = load_ground_truth(config.TRAIN_GT)
    print(f"Ground truth: {len(gt)} S1 entities")

    # Create train/val split
    train_ids, val_ids = create_train_val_split(gt)

    all_pairs = []
    all_s1_data = {}
    all_pool_data = {}

    # Process per country to save memory
    countries_seen = set()
    
    # Read S1 header to get countries
    s1_df = pd.read_csv(config.TRAIN_S1, sep="\t", dtype=str, keep_default_na=False)
    countries = s1_df["country"].unique()
    
    for country in countries:
        print(f"\n--- Processing {country} ---")
        
        # Filter S1 for this country
        s1_country = s1_df[s1_df["country"] == country]
        if sample_frac:
            s1_country = s1_country.sample(frac=sample_frac, random_state=42)
        
        # Load pool for this country
        s2_df = pd.read_csv(config.TRAIN_S2, sep="\t", dtype=str, keep_default_na=False)
        s3_df = pd.read_csv(config.TRAIN_S3, sep="\t", dtype=str, keep_default_na=False)
        pool_df = pd.concat([
            s2_df[s2_df["country"] == country],
            s3_df[s3_df["country"] == country],
        ], ignore_index=True)
        del s2_df, s3_df
        gc.collect()
        
        print(f"  S1: {len(s1_country)}, Pool: {len(pool_df)}")
        
        # Normalize
        s1_country = normalize_dataframe(s1_country.copy())
        pool_df = normalize_dataframe(pool_df.copy())
        
        # Blocking
        gt_country = {k: v for k, v in gt.items() if k in set(s1_country["entity_id"])}
        candidates = run_all_blocking(s1_country, pool_df, country=country, ground_truth=gt_country)
        
        # Build data dicts
        s1_data = build_record_dict(s1_country)
        pool_data = build_record_dict(pool_df)
        all_s1_data.update(s1_data)
        all_pool_data.update(pool_data)
        
        # Build pairs for train split only
        train_s1_country = train_ids & set(s1_country["entity_id"])
        pairs = build_training_pairs(train_s1_country, candidates, gt, neg_pos_ratio=5)
        all_pairs.extend(pairs)
        
        del s1_country, pool_df, candidates, s1_data, pool_data
        gc.collect()
    
    # Compute features
    print(f"\nComputing features for {len(all_pairs)} pairs...")
    X, y = compute_features_for_pairs(all_pairs, all_s1_data, all_pool_data)
    
    # Split features into train/val
    val_mask = np.array([p[0] in val_ids for p in all_pairs])
    train_mask = ~val_mask
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    val_pairs = [p for i, p in enumerate(all_pairs) if val_mask[i]]
    
    print(f"Train: {len(X_train)} pairs, Val: {len(X_val)} pairs")
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Threshold calibration
    best_t, best_f05, _ = find_optimal_threshold(model, X_val, val_pairs, gt)
    
    # Save
    save_model(model)
    threshold_path = os.path.join(config.MODEL_DIR, "threshold.txt")
    os.makedirs(os.path.dirname(threshold_path), exist_ok=True)
    with open(threshold_path, "w") as f:
        f.write(str(best_t))
    
    print(f"\nTRAINING COMPLETE: F0.5 = {best_f05:.4f}, threshold = {best_t:.2f}")
    return model, best_t


def run_predict(threshold: float = None):
    """Run prediction on test data."""
    if threshold is None:
        threshold_path = os.path.join(config.MODEL_DIR, "threshold.txt")
        if os.path.exists(threshold_path):
            with open(threshold_path) as f:
                threshold = float(f.read().strip())
        else:
            threshold = 0.5
    
    print(f"Using threshold: {threshold:.2f}")
    run_full_prediction(threshold=threshold)


def main():
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline")
    parser.add_argument("--mode", choices=["train", "predict", "sample", "eval"],
                        required=True, help="Pipeline mode")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold")
    parser.add_argument("--sample-size", type=int, default=5000, help="Sample size for test mode")
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Fraction of data for training (e.g. 0.1 for 10%%)")
    
    args = parser.parse_args()
    
    if args.mode == "sample":
        run_sample_test(n_sample=args.sample_size)
    elif args.mode == "train":
        run_train(sample_frac=args.sample_frac)
    elif args.mode == "predict":
        run_predict(threshold=args.threshold)
    elif args.mode == "eval":
        from .evaluate import evaluate_predictions_from_file
        score = evaluate_predictions_from_file(config.MATCHING_RESULTS, config.TRAIN_GT)
        print(f"F0.5 Score: {score:.4f}")


if __name__ == "__main__":
    main()
