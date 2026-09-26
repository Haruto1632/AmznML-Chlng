"""
name_cleaner.py — Business name normalisation rules.

Owner: Member A  |  Branch: feature/blocking

KEY DESIGN DECISION:
  Legal suffix tokens are EXCLUDED from the blocking token set because they
  are extremely high-frequency and create massive inverted index buckets.
  (e.g., "limited" appears in ~19% of Indian records, "incorporated" in ~9%)

  The normalized string still contains the suffix for feature computation.
  Only the blocking token set (business_name_tokens) excludes them.
"""

from __future__ import annotations

import re
import unicodedata
from typing import FrozenSet, Tuple

try:
    from unidecode import unidecode as _unidecode
    _HAS_UNIDECODE = True
except ImportError:
    _HAS_UNIDECODE = False

# Legal suffix normalisation map (for normalized string only)
# NOT used to generate blocking tokens.
LEGAL_SUFFIX_MAP: dict = {
    "corp": "corporation",
    "co": "company",
    "ltd": "limited",
    "pvt": "private",
    "inc": "incorporated",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "plc": "public limited company",
    "gmbh": "gesellschaft mit beschrankter haftung",
    "sarl": "societe a responsabilite limitee",
    "sas": "societe par actions simplifiee",
    "bv": "besloten vennootschap",
    "nv": "naamloze vennootschap",
    "ag": "aktiengesellschaft",
    "sa": "societe anonyme",
    "pty": "proprietary",
    "pte": "private",
    "kk": "kabushiki kaisha",
}

# All legal suffix tokens — original forms AND common expanded forms
# These are EXCLUDED from the BLOCKING token set.
# They are near-zero discriminative power for entity matching in blocking.
LEGAL_SUFFIX_TOKENS: frozenset = frozenset(
    list(LEGAL_SUFFIX_MAP.keys())
    + [
        # Expanded forms
        "corporation", "company", "limited", "private", "incorporated",
        "proprietary", "partnership", "liability",
        # Common expanded multi-word fragments (after splitting)
        "gesellschaft", "haftung", "beschrankter", "mit",
        "societe", "responsabilite", "limitee", "simplifiee", "anonyme",
        "besloten", "vennootschap", "naamloze", "aktiengesellschaft",
        "kabushiki", "kaisha", "public",
        # Short noise suffixes
        "ll", "lp",
    ]
)

# Stop-words removed from the blocking token SET.
# These are common function words and legal suffixes.
NAME_STOP_WORDS: frozenset = frozenset({
    "the", "a", "an", "of", "and", "or", "for", "in", "at",
    "by", "to", "on", "is", "its", "&",
}) | LEGAL_SUFFIX_TOKENS

# Additional data-driven stop words (discovered from analysis)
# High-frequency generic business terms that appear in >1% of records
DATA_STOP_WORDS: frozenset = frozenset({
    "services", "service", "group", "india", "enterprises", "enterprise",
    "solutions", "solution", "global", "international", "management",
    "trading", "holdings", "holding", "ventures", "venture",
    "industries", "industry", "technologies", "technology", "tech",
    "consultants", "consulting", "business", "resources",
    # Common Indian noisy transliterations of suffixes
    "limittedd", "praaivett", "praiveett", "praa", "limiited",
    # Common French generic terms
    "sarl", "eurl", "sas",
})

ALL_STOP_WORDS: frozenset = NAME_STOP_WORDS | DATA_STOP_WORDS

# Regex compiled once
_RE_WHITESPACE = re.compile(r"\s+")
_RE_PUNCT_FOR_TOKENS = re.compile(r"[^\w\s]")
_RE_URL = re.compile(r"(https?://\S+|www\.\S+|\S+\.(com|net|org|io|co|in|fr|uk|de|au)\b)", re.I)


def _transliterate(text: str) -> str:
    """Transliterate non-ASCII to ASCII if unidecode is available."""
    if _HAS_UNIDECODE:
        return _unidecode(text)
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def clean_name(name: str) -> Tuple[str, FrozenSet[str], str]:
    """
    Clean and normalise a business name.

    Parameters
    ----------
    name : str
        Raw business name (may be empty).

    Returns
    -------
    normalized : str
        Lightly normalised name (lowercase, whitespace, punctuation cleaned).
        Includes legal suffixes in their original form.
    tokens : frozenset[str]
        Set of DISCRIMINATIVE tokens for blocking.
        Excludes stop-words, legal suffixes, and data-derived generic terms.
        These are the tokens used for inverted index blocking.
    token_sorted : str
        Tokens sorted alphabetically and joined with space,
        for order-invariant comparison.
    """
    if not name or not isinstance(name, str):
        return "", frozenset(), ""

    # 1. Transliterate
    text = _transliterate(name)

    # 2. Lowercase
    text = text.lower()

    # 3. Remove URLs
    text = _RE_URL.sub(" ", text)

    # 4. & → and
    text = text.replace("&", " and ")

    # 5. Collapse whitespace
    text = _RE_WHITESPACE.sub(" ", text).strip()

    # 6. Remove punctuation for token purposes
    tokens_text = _RE_PUNCT_FOR_TOKENS.sub(" ", text)
    tokens_text = _RE_WHITESPACE.sub(" ", tokens_text).strip()

    # 7. Split into raw tokens
    raw_tokens = tokens_text.split()

    # 8. Filter to discriminative blocking tokens:
    #    - Length ≥ 3 (eliminate 2-char noise: "ll", "lp", etc.)
    #    - Not in ALL_STOP_WORDS (no legal suffixes, no generic terms)
    #    - Not pure digits
    meaningful_tokens = frozenset(
        t for t in raw_tokens
        if len(t) >= 3
        and t not in ALL_STOP_WORDS
        and not t.isdigit()
    )

    # 9. Token-sorted string for order-invariant matching
    token_sorted = " ".join(sorted(meaningful_tokens))

    return text, meaningful_tokens, token_sorted
