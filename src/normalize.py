"""
normalize.py — Text normalization for business entity resolution.

All functions are pure, stateless, and country-agnostic.
Handles: Indic scripts (8+ scripts via unidecode), French accents,
legal suffix expansion, noise token removal, domain name stripping,
address abbreviation expansion, landmark removal.

Design: normalize BEFORE any blocking or feature computation.
"""

import re
from unidecode import unidecode

from . import config


# ─── Precompile Regexes ──────────────────────────────────────────────────────

# Phone numbers: 10+ digit sequences (possibly with dashes/spaces)
_RE_PHONE = re.compile(r"(?<!\d)[\d\-\s]{10,}(?!\d)")

# ZIP/PIN codes: 5 or 6 digit sequences (US ZIP, India PIN, France postal)
_RE_ZIP = re.compile(r"\b(\d{5,6})\b")

# Numeric tokens in address (street numbers, plot numbers, house numbers)
_RE_NUMBERS = re.compile(r"\b(\d+)\b")

# Brackets: [Corp], (Limited), {xyz}
_RE_BRACKETS = re.compile(r"[\[\(\{]([^\]\)\}]*)[\]\)\}]")

# Landmark phrases: "near X Y", "opposite X Y", etc.
_RE_LANDMARK = re.compile(
    r"\b(near|opp|opposite|behind|beside|adjacent|next\s+to)\s+\S+(\s+\S+)?",
    re.IGNORECASE,
)

# Multiple whitespace
_RE_MULTI_SPACE = re.compile(r"\s+")

# Non-alphanumeric (keep spaces and digits)
_RE_NON_ALNUM = re.compile(r"[^a-z0-9\s]")

# Domain patterns: word.com, word.org, etc.
_RE_DOMAIN = re.compile(
    r"\b(\S+)\.(com|org|net|co\.in|in|co|io|biz|info|edu|gov|fr|us)\b",
    re.IGNORECASE,
)

# Ordinal suffixes: 1st, 2nd, 3rd, 45th, etc.
_RE_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)

# Noise tokens as a compiled set for O(1) lookup
_NOISE_LOWER = config.NOISE_TOKENS
_HONORIFICS = config.HONORIFICS
_NAME_STOPS = config.NAME_STOP_WORDS


def normalize_name(name: str) -> str:
    """
    Normalize a business name for blocking and feature computation.

    Pipeline (order matters):
    1. Transliterate non-Latin → Latin (unidecode)
    2. Strip domain suffixes (.com, .org, etc.) and split domain names
    3. Lowercase
    4. Remove bracket-wrapped text but keep the content inside
    5. Remove noise tokens (null, none, etc.)
    6. Remove phone numbers embedded in names
    7. & → and
    8. Remove honorifics (Mr, Smt, Shri, etc.)
    9. Expand legal suffixes (Corp → Corporation, etc.)
    10. Remove stop words
    11. Strip all non-alphanumeric (keep spaces)
    12. Collapse whitespace
    """
    if not name or not name.strip():
        return ""

    # 1. Transliterate ALL non-Latin scripts → Latin
    text = unidecode(name)

    # 2. Handle domain names: "maurewilliamscolombier.com" → "maurewilliamscolombier"
    #    Also split camelCase-like domains into words if detected
    text = _RE_DOMAIN.sub(r"\1", text)

    # 3. Lowercase
    text = text.lower().strip()

    # 4. Remove brackets but keep content: [Corp] → Corp, (Limited) → Limited
    text = _RE_BRACKETS.sub(r" \1 ", text)

    # 5. Remove noise tokens
    tokens = text.split()
    tokens = [t for t in tokens if t.lower().strip(".,;:-") not in _NOISE_LOWER]
    text = " ".join(tokens)

    # 6. Remove phone numbers
    text = _RE_PHONE.sub(" ", text)

    # 7. & → and
    text = text.replace("&", " and ")
    text = text.replace("+", " and ")

    # 8. Remove honorifics
    tokens = text.split()
    tokens = [t for t in tokens if t not in _HONORIFICS]
    text = " ".join(tokens)

    # 9. Expand legal suffixes
    tokens = text.split()
    expanded = []
    for token in tokens:
        clean = token.strip(".,;:-")
        if clean in config.LEGAL_SUFFIX_MAP:
            expanded.append(config.LEGAL_SUFFIX_MAP[clean])
        else:
            expanded.append(token)
    text = " ".join(expanded)

    # 10. Remove stop words
    tokens = text.split()
    tokens = [t for t in tokens if t not in _NAME_STOPS]
    text = " ".join(tokens)

    # 11. Strip all non-alphanumeric except spaces
    text = _RE_NON_ALNUM.sub(" ", text)

    # 12. Collapse whitespace
    text = _RE_MULTI_SPACE.sub(" ", text).strip()

    return text


def normalize_address(addr: str) -> str:
    """
    Normalize a business address for blocking and feature computation.

    Pipeline:
    1. Transliterate non-Latin → Latin (unidecode)
    2. Lowercase
    3. Remove noise tokens (null, <NULL>, none)
    4. Remove landmark phrases (near X, opposite Y)
    5. Normalize ordinals (45th → 45)
    6. Expand address abbreviations (St → Street, etc.)
    7. Strip all non-alphanumeric except spaces and digits
    8. Collapse whitespace
    """
    if not addr or not addr.strip():
        return ""

    # 1. Transliterate
    text = unidecode(addr)

    # 2. Lowercase
    text = text.lower().strip()

    # 3. Remove noise tokens (including <null>, <NULL>)
    text = re.sub(r"<[^>]*>", " ", text)  # Remove XML-like tags
    tokens = text.split()
    tokens = [t for t in tokens if t.strip(".,;:-") not in _NOISE_LOWER]
    text = " ".join(tokens)

    # 4. Remove landmark phrases
    text = _RE_LANDMARK.sub(" ", text)

    # 5. Normalize ordinals: 45th → 45, 1st → 1
    text = _RE_ORDINAL.sub(r"\1", text)

    # 6. Expand address abbreviations
    tokens = text.split()
    expanded = []
    for token in tokens:
        clean = token.strip(".,;:-#")
        if clean in config.ADDRESS_ABBREV_MAP:
            expanded.append(config.ADDRESS_ABBREV_MAP[clean])
        else:
            expanded.append(token)
    text = " ".join(expanded)

    # 7. Strip non-alphanumeric except spaces
    text = _RE_NON_ALNUM.sub(" ", text)

    # 8. Collapse whitespace
    text = _RE_MULTI_SPACE.sub(" ", text).strip()

    return text


def extract_zip(addr: str) -> str:
    """
    Extract ZIP/PIN/postal code from an address string.

    Handles:
    - US ZIP: 5 digits (optionally followed by -4 digits)
    - India PIN: 6 digits
    - France postal: 5 digits

    Returns the first match, or empty string if none found.
    """
    if not addr:
        return ""
    match = _RE_ZIP.search(addr)
    return match.group(1) if match else ""


def extract_numbers(text: str) -> set:
    """
    Extract all numeric substrings from text.
    Used for address number matching (street numbers, plot numbers, etc.)

    Returns a set of number strings for O(1) intersection.
    """
    if not text:
        return set()
    return set(_RE_NUMBERS.findall(text))


def make_name_tokens(normalized_name: str) -> set:
    """
    Split normalized name into a set of unique tokens.
    Used for token overlap blocking and Jaccard computation.
    """
    if not normalized_name:
        return set()
    return set(normalized_name.split())


def make_sorted_prefix(normalized_name: str, length: int = 8) -> str:
    """
    Create a prefix key from sorted name tokens.
    Used for prefix blocking (Strategy 6).

    Example: "golden one consultants private limited" → "consulta"
    (sorted: "consultants golden limited one private" → first 8 chars)
    """
    if not normalized_name:
        return ""
    tokens = sorted(normalized_name.split())
    joined = "".join(tokens)
    return joined[:length]


def char_trigrams(text: str) -> set:
    """
    Generate character trigrams from text.
    Used for char trigram Jaccard similarity feature.
    """
    if not text or len(text) < 3:
        return set()
    return {text[i : i + 3] for i in range(len(text) - 2)}
