"""
features.py — Pairwise feature engineering for entity resolution.

Computes 27 features for each (S1, candidate) pair.
All features are relational (compare two records), not absolute.
This ensures France works identically to US/India without retraining.

Uses: rapidfuzz (C++ backend, ~1M pairs/sec), jellyfish, numpy.
"""

import numpy as np
from rapidfuzz import fuzz
import jellyfish

from . import config
from .normalize import (
    normalize_name,
    normalize_address,
    extract_zip,
    extract_numbers,
    make_name_tokens,
    char_trigrams,
    make_sorted_prefix,
)


def _jaccard(set_a: set, set_b: set) -> float:
    """Token Jaccard similarity."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _overlap_coeff(set_a: set, set_b: set) -> float:
    """
    Overlap coefficient: |A∩B| / min(|A|, |B|).
    Perfect for truncated names (100% when one is a subset of another).
    """
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def _len_ratio(a: str, b: str) -> float:
    """Length ratio: min/max."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def compute_pair_features(
    norm_name_a: str,
    norm_name_b: str,
    norm_addr_a: str,
    norm_addr_b: str,
    raw_addr_a: str,
    raw_addr_b: str,
    emb_cosine: float,
    source_is_s3: bool,
) -> np.ndarray:
    """
    Compute all 27 features for a single (S1, candidate) pair.
    
    Args:
        norm_name_a: Normalized S1 business name
        norm_name_b: Normalized candidate business name
        norm_addr_a: Normalized S1 address
        norm_addr_b: Normalized candidate address
        raw_addr_a: Raw S1 address (for ZIP extraction)
        raw_addr_b: Raw candidate address (for ZIP extraction)
        emb_cosine: Precomputed cosine similarity of embeddings (or 0.0 if unavailable)
        source_is_s3: Whether candidate is from Source 3
    
    Returns:
        numpy array of 27 float32 features (order matches config.FEATURE_NAMES)
    """
    features = np.zeros(config.NUM_FEATURES, dtype=np.float32)
    
    # ─── Name tokens ──────────────────────────────────────────────────────
    name_tokens_a = make_name_tokens(norm_name_a)
    name_tokens_b = make_name_tokens(norm_name_b)
    
    # ─── Name features (indices 0-12) ─────────────────────────────────────
    
    # 0: name_jaro_winkler
    if norm_name_a and norm_name_b:
        features[0] = jellyfish.jaro_winkler_similarity(norm_name_a, norm_name_b)
    
    # 1: name_levenshtein_norm (1 - normalized distance)
    if norm_name_a and norm_name_b:
        features[1] = fuzz.ratio(norm_name_a, norm_name_b) / 100.0
    
    # 2: name_token_sort_ratio
    features[2] = fuzz.token_sort_ratio(norm_name_a, norm_name_b) / 100.0
    
    # 3: name_token_set_ratio
    features[3] = fuzz.token_set_ratio(norm_name_a, norm_name_b) / 100.0
    
    # 4: name_partial_ratio
    features[4] = fuzz.partial_ratio(norm_name_a, norm_name_b) / 100.0
    
    # 5: name_wratio (weighted best-of)
    features[5] = fuzz.WRatio(norm_name_a, norm_name_b) / 100.0
    
    # 6: name_jaccard
    features[6] = _jaccard(name_tokens_a, name_tokens_b)
    
    # 7: name_overlap_coeff
    features[7] = _overlap_coeff(name_tokens_a, name_tokens_b)
    
    # 8: name_char_trigram_jaccard
    tg_a = char_trigrams(norm_name_a)
    tg_b = char_trigrams(norm_name_b)
    features[8] = _jaccard(tg_a, tg_b)
    
    # 9: name_exact_match
    features[9] = 1.0 if norm_name_a == norm_name_b and norm_name_a else 0.0
    
    # 10: name_first_token_match
    first_a = norm_name_a.split()[0] if norm_name_a else ""
    first_b = norm_name_b.split()[0] if norm_name_b else ""
    features[10] = 1.0 if first_a and first_a == first_b else 0.0
    
    # 11: name_len_ratio
    features[11] = _len_ratio(norm_name_a, norm_name_b)
    
    # 12: name_common_token_count
    features[12] = float(len(name_tokens_a & name_tokens_b))
    
    # ─── Address features (indices 13-23) ─────────────────────────────────
    
    addr_tokens_a = set(norm_addr_a.split()) if norm_addr_a else set()
    addr_tokens_b = set(norm_addr_b.split()) if norm_addr_b else set()
    
    # 13: addr_jaro_winkler
    if norm_addr_a and norm_addr_b:
        features[13] = jellyfish.jaro_winkler_similarity(norm_addr_a, norm_addr_b)
    
    # 14: addr_token_sort_ratio
    features[14] = fuzz.token_sort_ratio(norm_addr_a, norm_addr_b) / 100.0
    
    # 15: addr_token_set_ratio
    features[15] = fuzz.token_set_ratio(norm_addr_a, norm_addr_b) / 100.0
    
    # 16: addr_partial_ratio
    features[16] = fuzz.partial_ratio(norm_addr_a, norm_addr_b) / 100.0
    
    # 17: addr_jaccard
    features[17] = _jaccard(addr_tokens_a, addr_tokens_b)
    
    # 18: addr_overlap_coeff
    features[18] = _overlap_coeff(addr_tokens_a, addr_tokens_b)
    
    # 19: addr_len_ratio
    features[19] = _len_ratio(norm_addr_a, norm_addr_b)
    
    # 20: addr_number_match (do numeric substrings match?)
    nums_a = extract_numbers(norm_addr_a)
    nums_b = extract_numbers(norm_addr_b)
    if nums_a and nums_b:
        features[20] = float(len(nums_a & nums_b) > 0)
    
    # 21: zip_match
    zip_a = extract_zip(raw_addr_a)
    zip_b = extract_zip(raw_addr_b)
    if zip_a and zip_b:
        features[21] = 1.0 if zip_a == zip_b else 0.0
    
    # 22: zip_present_both
    features[22] = 1.0 if zip_a and zip_b else 0.0
    
    # 23: addr_missing_either
    features[23] = 1.0 if (not norm_addr_a.strip() or not norm_addr_b.strip()) else 0.0
    
    # ─── Other features (indices 24-26) ───────────────────────────────────
    
    # 24: embedding_cosine
    features[24] = emb_cosine
    
    # 25: source_is_s3
    features[25] = 1.0 if source_is_s3 else 0.0
    
    # 26: name_has_common_prefix_5
    prefix_a = make_sorted_prefix(norm_name_a, length=5)
    prefix_b = make_sorted_prefix(norm_name_b, length=5)
    features[26] = 1.0 if prefix_a and prefix_a == prefix_b else 0.0
    
    return features


def compute_batch_features(
    norm_names_a: list,
    norm_names_b: list,
    norm_addrs_a: list,
    norm_addrs_b: list,
    raw_addrs_a: list,
    raw_addrs_b: list,
    emb_cosines: list,
    source_is_s3_flags: list,
) -> np.ndarray:
    """
    Compute features for a batch of pairs.
    
    Returns: (N, 27) float32 array.
    """
    n = len(norm_names_a)
    result = np.zeros((n, config.NUM_FEATURES), dtype=np.float32)
    
    for i in range(n):
        result[i] = compute_pair_features(
            norm_names_a[i],
            norm_names_b[i],
            norm_addrs_a[i],
            norm_addrs_b[i],
            raw_addrs_a[i],
            raw_addrs_b[i],
            emb_cosines[i] if emb_cosines else 0.0,
            source_is_s3_flags[i] if source_is_s3_flags else False,
        )
    
    return result
