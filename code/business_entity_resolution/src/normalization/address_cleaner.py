"""
address_cleaner.py — Business address normalisation rules.

Owner: Member A  |  Branch: feature/blocking

French address terms are kept as-is (rue, avenue, cedex etc.).
Address numbers extracted into a separate list for blocking.
"""

from __future__ import annotations

import re
from typing import List, Tuple

try:
    from unidecode import unidecode as _unidecode
    _HAS_UNIDECODE = True
except ImportError:
    _HAS_UNIDECODE = False

# English/common address abbreviation expansion.
# French terms: rue, avenue, boulevard, cedex, arrondissement → kept unchanged.
# Indian terms: nagar, marg, colony → kept unchanged.
# Only expand English abbreviations that are clearly directional or road-type.
ADDRESS_ABBREV_MAP: dict[str, str] = {
    "st": "street",
    "rd": "road",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "hwy": "highway",
    "fwy": "freeway",
    "pkwy": "parkway",
    "expy": "expressway",
    "ste": "suite",
    "apt": "apartment",
    "fl": "floor",
    # directionals (short)
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}

# Regex patterns (compiled once)
_RE_WHITESPACE = re.compile(r"\s+")
_RE_PUNCT = re.compile(r"[^\w\s\-]")
# Extract numeric sequences including hyphenated (ZIP+4: 12345-6789)
_RE_NUMBERS = re.compile(r"\b\d+(?:-\d+)?\b")
# Words for token extraction (alpha only, length ≥ 2)
_RE_ALPHA_TOKEN = re.compile(r"\b[a-z]{2,}\b")


def clean_address(address: str) -> Tuple[str, List[str]]:
    """
    Clean and normalise a business address.

    Parameters
    ----------
    address : str
        Raw business address (may be empty).

    Returns
    -------
    normalized : str
        Normalised address string (lowercase, whitespace collapsed,
        abbreviations expanded where known).
    numbers : list[str]
        Extracted numeric tokens (building numbers, PIN codes, ZIP codes).
        Used for address blocking.
    """
    if not address or not isinstance(address, str):
        return "", []

    # 1. Transliterate if needed
    if _HAS_UNIDECODE:
        text = _unidecode(address)
    else:
        text = address

    # 2. Lowercase
    text = text.lower()

    # 3. Extract numbers BEFORE removing punctuation (preserves hyphenated ZIPs)
    numbers = _RE_NUMBERS.findall(text)

    # 4. Remove punctuation except hyphens (for hyphenated ZIP codes already captured)
    text = _RE_PUNCT.sub(" ", text)

    # 5. Collapse whitespace
    text = _RE_WHITESPACE.sub(" ", text).strip()

    # 6. Expand abbreviations token-by-token (only exact word boundary matches)
    tokens = text.split()
    expanded = []
    for tok in tokens:
        expanded.append(ADDRESS_ABBREV_MAP.get(tok, tok))
    text = " ".join(expanded)

    return text, numbers


def extract_address_tokens(normalized_address: str) -> frozenset:
    """
    Extract alphabetic tokens from a normalized address for blocking.
    Returns a frozenset of lowercase alpha tokens (length ≥ 2).
    Excludes very common noise words.
    """
    _ADDR_STOP = frozenset({"street", "road", "avenue", "drive", "lane",
                            "place", "court", "north", "south", "east", "west",
                            "suite", "floor", "apartment"})
    tokens = frozenset(_RE_ALPHA_TOKEN.findall(normalized_address))
    return tokens - _ADDR_STOP
