"""
phonetic_blocking.py — Soundex/Metaphone phonetic blocking.

Owner: Member A  |  Branch: feature/blocking

Strategy: phonetic
  - Applies Soundex to the FIRST alphabetic token of the normalized business name
  - Only for ASCII-safe tokens (skips Devanagari, Chinese, etc.)
  - Combined with country to reduce false positives
  - Uses jellyfish.soundex (fast C implementation)

Design notes:
  - Applied on first meaningful token only (most discriminative for business names)
  - Phonetic codes have ~4-char resolution; combined with country keeps precision high
  - Non-ASCII names are automatically skipped (no crash, just no phonetic candidate)
  - This mainly catches: spelling variations, typos, transliteration variants
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Set, Tuple

import pandas as pd

try:
    import jellyfish
    _HAS_JELLYFISH = True
except ImportError:
    jellyfish = None
    _HAS_JELLYFISH = False

from ...shared.schemas import (
    COL_ENTITY_ID,
    COL_NAME_TOKENS,
    COL_COUNTRY_NORMALIZED,
    COL_SOURCE1_ID,
    COL_CANDIDATE_ID,
    STRATEGY_PHONETIC,
)

logger = logging.getLogger(__name__)


def _soundex_safe(token: str) -> str | None:
    """
    Return Soundex code for token, or None if:
    - token is not ASCII
    - jellyfish is not available
    - any error occurs
    """
    if not _HAS_JELLYFISH:
        return None
    if not token.isascii() or not token.isalpha():
        return None
    try:
        return jellyfish.soundex(token)
    except Exception:
        return None


def _pick_phonetic_token(tokens: frozenset) -> str | None:
    """
    Pick the single best token to use for phonetic blocking.
    Prefers the longest ASCII alphabetic token (more discriminative).
    Returns None if no suitable token found.
    """
    candidates = [t for t in tokens if t.isascii() and t.isalpha() and len(t) >= 2]
    if not candidates:
        return None
    return max(candidates, key=len)


def build_phonetic_index(
    candidates: pd.DataFrame,
) -> Dict[Tuple[str, str], Set[str]]:
    """
    Build phonetic index: (country, soundex_code) → set(entity_ids).

    Parameters
    ----------
    candidates : pd.DataFrame
        Combined S2+S3 with columns: entity_id, business_name_tokens,
        country_normalized.

    Returns
    -------
    dict mapping (country, soundex_code) → set of entity_ids
    """
    if not _HAS_JELLYFISH:
        logger.warning("jellyfish not available; phonetic blocking disabled.")
        return {}

    index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    skipped = 0

    for row in candidates.itertuples(index=False):
        tokens = row.business_name_tokens
        country = row.country_normalized
        eid = row.entity_id

        tok = _pick_phonetic_token(tokens)
        if tok is None:
            skipped += 1
            continue

        code = _soundex_safe(tok)
        if code is None:
            skipped += 1
            continue

        index[(country, code)].add(eid)

    logger.info(
        f"Phonetic index: {len(index):,} (country, code) keys; "
        f"{skipped:,} records skipped (non-ASCII/no suitable token)"
    )
    return dict(index)


def generate_phonetic_pairs(
    source1: pd.DataFrame,
    phonetic_index: Dict[Tuple[str, str], Set[str]],
) -> pd.DataFrame:
    """
    Strategy: phonetic
    Generates pairs where S1 and candidate have matching Soundex code
    for their primary token AND the same country.

    Returns DataFrame: source1_entity_id, candidate_entity_id
    """
    if not _HAS_JELLYFISH or not phonetic_index:
        logger.warning("Phonetic blocking skipped (jellyfish unavailable or empty index).")
        return pd.DataFrame(columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])

    records = []
    for row in source1.itertuples(index=False):
        s1_id = row.entity_id
        tokens = row.business_name_tokens
        country = row.country_normalized

        tok = _pick_phonetic_token(tokens)
        if tok is None:
            continue

        code = _soundex_safe(tok)
        if code is None:
            continue

        for cand_id in phonetic_index.get((country, code), ()):
            records.append((s1_id, cand_id))

    if not records:
        return pd.DataFrame(columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])

    return pd.DataFrame(records, columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])
