"""
name_cleaner.py — Business name normalisation rules.

Owner: Member A  |  Branch: feature/blocking

Vectorized implementation using pandas .str operations where possible.
Row-level clean_name() is used for per-record processing.
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

# Legal suffix normalisation map (expand to canonical long form for blocking)
# Keys are the short forms that appear in raw data.
LEGAL_SUFFIX_MAP: dict[str, str] = {
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
    "plc": "public limited company",
    "kk": "kabushiki kaisha",
}

# Stop-words removed only from the token SET (not from the normalized string).
# Keep short but effective: these tokens add near-zero discriminative value.
NAME_STOP_WORDS: frozenset = frozenset({
    "the", "a", "an", "of", "and", "or", "for", "in", "at",
    "by", "to", "on", "is", "its", "&",
})

# Regex compiled once
_RE_WHITESPACE = re.compile(r"\s+")
_RE_PUNCT_FOR_TOKENS = re.compile(r"[^\w\s]")
_RE_URL = re.compile(r"(https?://\S+|www\.\S+|\S+\.(com|net|org|io|co|in|fr|uk|de|au)\b)", re.I)


def _transliterate(text: str) -> str:
    """Transliterate non-ASCII to ASCII if unidecode is available."""
    if _HAS_UNIDECODE:
        return _unidecode(text)
    # Fallback: NFD decompose and strip combining chars
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
        Originals are NOT destroyed — this is a 'heavy' normalized form.
    tokens : frozenset[str]
        Set of meaningful tokens (stop-words removed, length ≥ 2).
    token_sorted : str
        Tokens sorted alphabetically and joined with space,
        for order-invariant comparison.
    """
    if not name or not isinstance(name, str):
        return "", frozenset(), ""

    # 1. Transliterate (Devanagari, Cyrillic etc. → ASCII approximation)
    text = _transliterate(name)

    # 2. Lowercase
    text = text.lower()

    # 3. Remove URLs
    text = _RE_URL.sub(" ", text)

    # 4. & → and
    text = text.replace("&", " and ")

    # 5. Collapse whitespace
    text = _RE_WHITESPACE.sub(" ", text).strip()

    # 6. Remove punctuation for token purposes (keep for normalized string)
    tokens_text = _RE_PUNCT_FOR_TOKENS.sub(" ", text)
    tokens_text = _RE_WHITESPACE.sub(" ", tokens_text).strip()

    # 7. Split into raw tokens
    raw_tokens = tokens_text.split()

    # 8. Expand legal suffixes
    expanded_tokens = []
    for tok in raw_tokens:
        expanded_tokens.append(LEGAL_SUFFIX_MAP.get(tok, tok))

    # 9. Remove stop-words; keep tokens with length ≥ 2
    meaningful_tokens = frozenset(
        t for t in expanded_tokens
        if t and len(t) >= 2 and t not in NAME_STOP_WORDS
    )

    # 10. Token-sorted string for order-invariant matching
    token_sorted = " ".join(sorted(meaningful_tokens))

    return text, meaningful_tokens, token_sorted
