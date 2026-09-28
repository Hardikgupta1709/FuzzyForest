"""
train.py — LightGBM training for entity resolution.

Training data construction:
- Split by S1 entity (not by pair) to avoid leakage
- All positives from ground truth within candidate set
- 5× hard negatives per positive from blocking candidates
- float32 features for memory efficiency

Model: LightGBM (MIT license, ~KB trained model)
"""

import os
import csv
import random
import numpy as np
import lightgbm as lgb
from tqdm import tqdm

from . import config
from .features import compute_pair_features
from .evaluate import macro_f05, load_ground_truth


def create_train_val_split(ground_truth: dict, val_fraction: float = 0.15, seed: int = 42):
    """
    Split S1 entity IDs into train/val sets.
    Stratified by having-matches vs singleton.
    
    Returns: (train_s1_ids, val_s1_ids)
    """
    random.seed(seed)
    
    singletons = [s1 for s1, matches in ground_truth.items() if not matches]
    non_singletons = [s1 for s1, matches in ground_truth.items() if matches]
    
    random.shuffle(singletons)
    random.shuffle(non_singletons)
    
    val_n_single = int(len(singletons) * val_fraction)
    val_n_non = int(len(non_singletons) * val_fraction)
    
    val_ids = set(singletons[:val_n_single]) | set(non_singletons[:val_n_non])
    train_ids = set(singletons[val_n_single:]) | set(non_singletons[val_n_non:])
    
    print(f"Train: {len(train_ids)} S1 entities ({len(train_ids) - len(singletons) + val_n_single} non-singleton)")
    print(f"Val:   {len(val_ids)} S1 entities ({val_n_non} non-singleton)")
    
    return train_ids, val_ids


def build_training_pairs(
    train_s1_ids: set,
    candidates: dict,
    ground_truth: dict,
    neg_pos_ratio: int = 5,
    seed: int = 42,
):
    """
    Build (s1_id, candidate_id, label) training pairs.
    
    For each S1 entity in train set:
    - All positives: ground truth matches that appear in candidates
    - Hard negatives: random sample from remaining candidates (5× per positive)
    
    Returns: list of (s1_id, cand_id, label) tuples
    """
    random.seed(seed)
    pairs = []
    
    missed_positives = 0
    total_positives = 0
    
    for s1_id in tqdm(train_s1_ids, desc="Building pairs"):
        true_matches = ground_truth.get(s1_id, set())
        cand_set = candidates.get(s1_id, set())
        
        if not cand_set:
            continue
        
        # Positives: ground truth matches found in candidates
        positives = true_matches & cand_set
        missed = true_matches - cand_set
        missed_positives += len(missed)
        total_positives += len(true_matches)
        
        for cand_id in positives:
            pairs.append((s1_id, cand_id, 1))
        
        # Hard negatives: candidates that are NOT true matches
        negatives = list(cand_set - true_matches)
        n_neg = min(len(negatives), len(positives) * neg_pos_ratio)
        if n_neg > 0:
            sampled_neg = random.sample(negatives, n_neg)
            for cand_id in sampled_neg:
                pairs.append((s1_id, cand_id, 0))
    
    n_pos = sum(1 for _, _, l in pairs if l == 1)
    n_neg = sum(1 for _, _, l in pairs if l == 0)
    print(f"Training pairs: {len(pairs)} ({n_pos} positive, {n_neg} negative)")
    print(f"Blocking missed {missed_positives}/{total_positives} positives ({100*missed_positives/max(total_positives,1):.2f}%)")
    
    return pairs


def compute_features_for_pairs(
    pairs: list,
    s1_data: dict,
    pool_data: dict,
    embeddings_s1: dict = None,
    embeddings_pool: dict = None,
):
    """
    Compute feature matrix for training pairs.
    
    Args:
        pairs: list of (s1_id, cand_id, label)
        s1_data: {entity_id: {'norm_name', 'norm_addr', 'raw_addr'}}
        pool_data: {entity_id: {'norm_name', 'norm_addr', 'raw_addr'}}
        embeddings_s1: {entity_id: np.array} or None
        embeddings_pool: {entity_id: np.array} or None
    
    Returns: (X, y) where X is (N, 27) float32 array, y is (N,) int array
    """
    n = len(pairs)
    X = np.zeros((n, config.NUM_FEATURES), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    
    for i, (s1_id, cand_id, label) in enumerate(tqdm(pairs, desc="Computing features")):
        s1 = s1_data.get(s1_id, {})
        cand = pool_data.get(cand_id, {})
        
        # Embedding cosine
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
        y[i] = label
    
    return X, y


def train_model(X_train, y_train, X_val, y_val, params=None):
    """
    Train LightGBM binary classifier.
    
    Returns: trained lgb.Booster
    """
    if params is None:
        params = config.LGBM_PARAMS.copy()
    
    n_estimators = params.pop("n_estimators", 1000)
    early_stopping = params.pop("early_stopping_rounds", 50)
    
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=config.FEATURE_NAMES)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=config.FEATURE_NAMES, reference=train_set)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping),
        lgb.log_evaluation(period=50),
    ]
    
    model = lgb.train(
        params,
        train_set,
        num_boost_round=n_estimators,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    
    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(config.FEATURE_NAMES, importance), key=lambda x: -x[1])
    print("\nTop 10 features by gain:")
    for fname, imp in feat_imp[:10]:
        print(f"  {fname}: {imp:.1f}")
    
    return model


def find_optimal_threshold(
    model: lgb.Booster,
    X_val: np.ndarray,
    val_pairs: list,
    ground_truth: dict,
    sweep_min: float = 0.30,
    sweep_max: float = 0.95,
    sweep_step: float = 0.01,
):
    """
    Sweep threshold to maximize F0.5 on validation split.
    
    Returns: (best_threshold, best_f05, all_results)
    """
    # Get predictions
    probas = model.predict(X_val)
    
    # Group by S1 entity
    s1_candidates = {}  # {s1_id: {cand_id: probability}}
    for (s1_id, cand_id, _), prob in zip(val_pairs, probas):
        if s1_id not in s1_candidates:
            s1_candidates[s1_id] = {}
        s1_candidates[s1_id][cand_id] = prob
    
    # Get validation S1 IDs from pairs
    val_s1_ids = set(s1_id for s1_id, _, _ in val_pairs)
    val_gt = {s1_id: ground_truth.get(s1_id, set()) for s1_id in val_s1_ids}
    
    # Also include S1 entities with no candidates (singletons)
    for s1_id in ground_truth:
        if s1_id not in val_gt and s1_id in val_s1_ids:
            val_gt[s1_id] = ground_truth[s1_id]
    
    best_threshold = 0.5
    best_f05 = 0.0
    results = []
    
    thresholds = np.arange(sweep_min, sweep_max + sweep_step, sweep_step)
    
    for threshold in thresholds:
        predictions = {}
        for s1_id in val_gt:
            if s1_id in s1_candidates:
                matched = {cid for cid, prob in s1_candidates[s1_id].items() if prob >= threshold}
                predictions[s1_id] = matched
            else:
                predictions[s1_id] = set()
        
        f05 = macro_f05(predictions, val_gt)
        results.append((threshold, f05))
        
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = threshold
    
    print(f"\nThreshold sweep results:")
    print(f"  Best threshold: {best_threshold:.2f}")
    print(f"  Best F0.5:      {best_f05:.4f}")
    
    # Print neighborhood around best
    for t, f in results:
        if abs(t - best_threshold) <= 0.05:
            marker = " <<<" if abs(t - best_threshold) < 0.001 else ""
            print(f"  t={t:.2f}  F0.5={f:.4f}{marker}")
    
    return best_threshold, best_f05, results


def save_model(model: lgb.Booster, path: str = None):
    """Save trained model."""
    if path is None:
        path = config.MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)
    print(f"Model saved to {path}")


def load_model(path: str = None) -> lgb.Booster:
    """Load trained model."""
    if path is None:
        path = config.MODEL_PATH
    model = lgb.Booster(model_file=path)
    print(f"Model loaded from {path}")
    return model
