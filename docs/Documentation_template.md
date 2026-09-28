# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Solo  
**Team Members:** Hardik  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

Two-stage entity resolution pipeline combining multi-strategy blocking (6 strategies) with a LightGBM binary classifier on 27 handcrafted pairwise similarity features. Key innovations: (1) character n-gram TF-IDF blocking for typo robustness, (2) address-based blocking to catch DBA/trade name aliases, and (3) overlap coefficient feature for truncated name matching. Achieves 99.7%+ blocking recall and 0.99+ F0.5 on validation.

---

## 2. Methodology

### 2.1 Problem Analysis

Key insights discovered during EDA:
- **Zero cross-country matches** in training data (7.6M ground truth pairs checked) — enables safe country partitioning
- **8+ Indic scripts** in S2/S3 (Hindi, Bengali, Gujarati, Tamil, Telugu, Kannada, Malayalam, Punjabi) — handled via `unidecode` transliteration
- **2.07% of matches** have name similarity < 0.3 after transliteration — these are DBA/trade name aliases (e.g., "Harris Tankers" ↔ "Miraonyx") that share addresses
- **98.4% of DBA cases** have similar addresses — validates address-based blocking as critical
- **~3% of S2/S3** have missing addresses — requires name-only fallback
- **France is test-only** (zero in training) — all features must be relational/country-agnostic
- **Noise tokens**: `null`, `<NULL>`, phone numbers in names, `HN`, `DOOR NO`, brackets, honorifics

### 2.2 Solution Strategy

**Approach Type:** Blocking + Gradient Boosted Decision Tree Classifier  
**Core Innovation:** Union of 6 complementary blocking strategies (exact name, token overlap, char n-gram TF-IDF, FAISS ANN embeddings, address ZIP+token overlap, prefix) achieves 99.7%+ recall while reducing comparison space by 99.99%

---

## 3. Candidate Generation (Blocking)

Six complementary strategies, unioned per S1 entity:

1. **Exact Normalized Name** — Hash lookup after unidecode + lowercase + legal suffix expansion
2. **Token Overlap (Jaccard ≥ 0.30)** — Inverted index with frequency filtering (skip tokens appearing in >10K records)
3. **Character N-gram TF-IDF (cosine ≥ 0.20)** — `char_wb` analyzer with (3,4)-grams; typo-robust (a typo corrupts only 3 trigrams vs destroying entire word in word-level TF-IDF)
4. **Address Blocking** — ZIP/PIN match + ≥3 address token overlap; catches DBA/trade name cases where names are completely different but addresses match
5. **Prefix Blocking** — First 8 chars of sorted normalized name tokens
6. **FAISS ANN (top-30)** — E5-small multilingual embeddings with IVF index (on SageMaker)

- **Blocking keys used:** Normalized name hash, name token inverted index, char n-gram TF-IDF, address tokens + ZIP codes, name prefix, semantic embeddings
- **Candidate pairs generated:** ~50 per S1 entity (union of all strategies)
- **How true matches preserved:** 99.74% blocking recall measured on validation; multiple redundant strategies ensure that if one strategy misses a pair, another catches it

---

## 4. Matching Model

**Features used (27 total):**
- Name features (13): Jaro-Winkler, normalized Levenshtein, token_sort_ratio, token_set_ratio, partial_ratio, WRatio, Jaccard, overlap coefficient, char trigram Jaccard, exact match, first token match, length ratio, common token count
- Address features (11): Jaro-Winkler, token_sort_ratio, token_set_ratio, partial_ratio, Jaccard, overlap coefficient, length ratio, number match, ZIP match, ZIP present both, missing either
- Other (3): embedding cosine similarity, source_is_s3, common prefix 5

**Model type:** LightGBM (MIT license, <8B params)  
- `num_leaves=63`, `learning_rate=0.05`, `feature_fraction=0.8`
- Early stopping on validation binary_logloss

**Threshold selection method:** Grid sweep over [0.30, 0.95] step 0.01, maximizing macro F0.5 on 15% entity-level held-out validation set

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9986 on 2000-entity targeted validation sample
- **Top features by importance:** addr_token_set_ratio (135K gain), addr_jaccard (100K), name_jaro_winkler (26K), addr_jaro_winkler (13K), name_wratio (8K)
- **Common false positives (wrong merges):** Businesses at same address with similar but different names (e.g., two restaurants at same strip mall)
- **Common false negatives (missed matches):** Completely synthetic alias names with abbreviated/truncated addresses; transliteration failures for rare Indic scripts

---

## 6. Conclusion

This solution demonstrates that classical string similarity features combined with gradient boosted trees remain highly effective for entity resolution at scale. The key insight is that blocking recall is the true bottleneck — investing in 6 complementary blocking strategies (including the critical address-based blocking for DBA/trade names) yields near-perfect recall, allowing the precision-optimized LightGBM classifier to achieve excellent F0.5 scores.

---

## Appendix

### A. Code Artefacts

```
code/business_entity_resolution/
├── src/
│   ├── config.py        # Central configuration
│   ├── normalize.py     # Text normalization (unidecode, legal suffixes)
│   ├── blocking.py      # 6 blocking strategies
│   ├── features.py      # 27 pairwise features
│   ├── train.py         # LightGBM training + threshold calibration
│   ├── predict.py       # Inference pipeline
│   ├── evaluate.py      # F0.5 scorer
│   └── pipeline.py      # CLI orchestrator
├── README.md
└── requirements.txt
```

Entry point: `python -m src.pipeline --mode predict` (from `code/business_entity_resolution/`)

### B. Additional Results

- Blocking recall: 99.74% (only 0.20% of true matches missed by blocking)
- Average candidates per S1 entity: ~50 (99.99% reduction from 10M pool)
- LightGBM training: 232 boosting rounds, val logloss 0.000615
- Zero cross-country matches verified on all 7.6M ground truth pairs

---

**Note:** All models used are MIT/Apache 2.0 licensed and under 8B parameters. No external data lookups are performed.
