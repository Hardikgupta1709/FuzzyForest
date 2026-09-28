"""
config.py — Central configuration for the entity resolution pipeline.

All paths, hyperparameters, abbreviation dictionaries, and regex patterns.
Country-agnostic: no hard-coded country lists or one-hot encodings.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

MATCHING_RESULTS = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model.txt")

# ─── Blocking Hyperparameters ─────────────────────────────────────────────────
TOKEN_JACCARD_THRESHOLD = 0.30       # Strategy 2: name token overlap
CHAR_TFIDF_THRESHOLD = 0.20         # Strategy 3: char n-gram TF-IDF cosine
FAISS_TOP_K = 30                     # Strategy 4: ANN nearest neighbors
ADDR_TOKEN_OVERLAP_MIN = 3          # Strategy 5: address token overlap (with ZIP)
ADDR_TOKEN_OVERLAP_NO_ZIP = 4       # Strategy 5: address token overlap (without ZIP)
PREFIX_BLOCK_LEN = 8                 # Strategy 6: first-N-chars of sorted name tokens

# ─── TF-IDF Hyperparameters ──────────────────────────────────────────────────
TFIDF_ANALYZER = "char_wb"           # Character n-grams with word boundaries
TFIDF_NGRAM_RANGE = (3, 4)          # Trigrams + 4-grams
TFIDF_MAX_FEATURES = 200_000        # Vocabulary cap to control memory

# ─── FAISS Hyperparameters ───────────────────────────────────────────────────
FAISS_NLIST = 4096                   # Number of IVF cells
FAISS_NPROBE = 128                   # Cells to probe at query time

# ─── Embedding Model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
EMBEDDING_PREFIX = "passage: "       # E5 models require this prefix

# ─── LightGBM Hyperparameters ────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 100,
    "scale_pos_weight": 5,
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "num_threads": 8,
    "seed": 42,
}

# ─── Training ────────────────────────────────────────────────────────────────
VAL_FRACTION = 0.15                  # 15% of S1 entities for validation
NEG_POS_RATIO = 5                    # Hard negatives per positive
RANDOM_SEED = 42

# ─── Threshold ───────────────────────────────────────────────────────────────
THRESHOLD_SWEEP_MIN = 0.30
THRESHOLD_SWEEP_MAX = 0.95
THRESHOLD_SWEEP_STEP = 0.01

# ─── Legal Suffix Expansion Dictionary ───────────────────────────────────────
# Covers US, India, and France legal entities.
# Maps abbreviated form → canonical expanded form.
LEGAL_SUFFIX_MAP = {
    # US
    "corp": "corporation",
    "inc": "incorporated",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "co": "company",
    "assoc": "associates",
    "assn": "association",
    "intl": "international",
    "natl": "national",
    "svcs": "services",
    "svc": "service",
    "mfg": "manufacturing",
    "engr": "engineering",
    "engg": "engineering",
    "mgmt": "management",
    "grp": "group",
    "hldgs": "holdings",
    # India
    "pvt": "private",
    "ltd": "limited",
    "priv": "private",
    "lim": "limited",
    "bros": "brothers",
    "govt": "government",
    "pvt.": "private",
    "ltd.": "limited",
    # France
    "sarl": "societe a responsabilite limitee",
    "sas": "societe par actions simplifiee",
    "sa": "societe anonyme",
    "sci": "societe civile immobiliere",
    "eurl": "entreprise unipersonnelle a responsabilite limitee",
    "snc": "societe en nom collectif",
    "scp": "societe civile professionnelle",
    "ste": "societe",
}

# ─── Address Abbreviation Expansion ─────────────────────────────────────────
ADDRESS_ABBREV_MAP = {
    "st": "street",
    "st.": "street",
    "rd": "road",
    "rd.": "road",
    "ave": "avenue",
    "ave.": "avenue",
    "blvd": "boulevard",
    "blvd.": "boulevard",
    "dr": "drive",
    "dr.": "drive",
    "ct": "court",
    "ct.": "court",
    "ln": "lane",
    "ln.": "lane",
    "hwy": "highway",
    "hwy.": "highway",
    "apt": "apartment",
    "apt.": "apartment",
    "ste": "suite",
    "ste.": "suite",
    "fl": "floor",
    "fl.": "floor",
    "bldg": "building",
    "bldg.": "building",
    "pkwy": "parkway",
    "cir": "circle",
    "pl": "place",
    "sq": "square",
    "trl": "trail",
    "ter": "terrace",
    "expy": "expressway",
    "hn": "house number",
    "h.no": "house number",
    "h.no.": "house number",
    "no": "number",
    "no.": "number",
}

# ─── Honorifics / Titles to Strip ───────────────────────────────────────────
HONORIFICS = {
    "mr", "mrs", "ms", "dr", "prof", "smt", "shri", "sri", "kumari",
    "thiru", "selvi", "srimati",
}

# ─── Noise Tokens to Remove ─────────────────────────────────────────────────
NOISE_TOKENS = {
    "null", "none", "na", "n/a", "nil", "undefined", "nan",
}

# ─── Stop Words to Remove from Names ────────────────────────────────────────
NAME_STOP_WORDS = {
    "the", "of", "and", "a", "an", "for", "in", "at", "to", "by",
    "its", "or", "on", "with",
}

# ─── Domain Suffixes to Strip ───────────────────────────────────────────────
DOMAIN_SUFFIXES = {
    ".com", ".org", ".net", ".co.in", ".in", ".co", ".io", ".biz",
    ".info", ".edu", ".gov", ".fr", ".us",
}

# ─── Feature Names (ordered) ────────────────────────────────────────────────
FEATURE_NAMES = [
    "name_jaro_winkler",
    "name_levenshtein_norm",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_wratio",
    "name_jaccard",
    "name_overlap_coeff",
    "name_char_trigram_jaccard",
    "name_exact_match",
    "name_first_token_match",
    "name_len_ratio",
    "name_common_token_count",
    "addr_jaro_winkler",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_partial_ratio",
    "addr_jaccard",
    "addr_overlap_coeff",
    "addr_len_ratio",
    "addr_number_match",
    "zip_match",
    "zip_present_both",
    "addr_missing_either",
    "embedding_cosine",
    "source_is_s3",
    "name_has_common_prefix_5",
]

NUM_FEATURES = len(FEATURE_NAMES)  # 27
