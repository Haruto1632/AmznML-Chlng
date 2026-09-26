"""
ngram_blocking.py — Character n-gram MinHash/LSH blocking.

Owner: Member A  |  Branch: feature/blocking

Strategy: char_ngram_lsh
  - Extracts character n-grams (default n=3) from normalized business name
  - Uses datasketch MinHash LSH for approximate Jaccard similarity search
  - All S2+S3 candidates inserted into LSH; S1 queries it
  - Memory-conscious: LSH uses band+bucket structure, not dense matrix

Scalability:
  - datasketch MinHash LSH is O(n * num_perm) to build
  - Query: O(k * num_perm) per S1 entity where k is band count
  - NO dense similarity matrix created anywhere
  - For 7.5M candidates: ~128 hash funcs × 7.5M = manageable in RAM

Design choices:
  - n=3 char n-grams: good coverage for business names
  - threshold=0.25: low Jaccard threshold for high recall at blocking stage
  - num_perm=64: lighter than 128 for speed; can tune
  - Applied on normalized name (post-transliteration)

Note: datasketch inserts require unique string keys.
"""

from __future__ import annotations

import logging
from typing import Set, Tuple

import pandas as pd

try:
    from datasketch import MinHash, MinHashLSH
    _HAS_DATASKETCH = True
except ImportError:
    MinHash = MinHashLSH = None
    _HAS_DATASKETCH = False

from ...shared.schemas import (
    COL_ENTITY_ID,
    COL_NAME_NORMALIZED,
    COL_SOURCE1_ID,
    COL_CANDIDATE_ID,
    STRATEGY_NGRAM_LSH,
)

logger = logging.getLogger(__name__)


def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _make_minhash(ngrams: Set[str], num_perm: int) -> "MinHash":
    """Create a MinHash signature from a set of n-grams."""
    m = MinHash(num_perm=num_perm)
    for ng in ngrams:
        m.update(ng.encode("utf-8"))
    return m


def build_ngram_lsh(
    candidates: pd.DataFrame,
    num_perm: int = 64,
    threshold: float = 0.25,
    ngram_n: int = 3,
) -> Tuple["MinHashLSH", dict]:
    """
    Build MinHash LSH index over all S2+S3 candidates.

    Parameters
    ----------
    candidates : pd.DataFrame
        S2+S3 with columns: entity_id, business_name_normalized
    num_perm : int
        Number of MinHash permutations (higher = more accurate, more RAM)
    threshold : float
        Jaccard similarity threshold for LSH bucketing
    ngram_n : int
        Character n-gram size

    Returns
    -------
    lsh : MinHashLSH
        Populated LSH index
    minhash_cache : dict
        entity_id → MinHash (used to look up hashes if needed)
    """
    if not _HAS_DATASKETCH:
        logger.warning("datasketch not available; n-gram LSH blocking disabled.")
        return None, {}

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhash_cache = {}
    skipped = 0
    inserted = 0

    for row in candidates.itertuples(index=False):
        eid = row.entity_id
        name = row.business_name_normalized

        if not name:
            skipped += 1
            continue

        ngrams = _char_ngrams(name, ngram_n)
        if not ngrams:
            skipped += 1
            continue

        m = _make_minhash(ngrams, num_perm)
        try:
            lsh.insert(eid, m)
            minhash_cache[eid] = m
            inserted += 1
        except ValueError:
            # Duplicate key — skip silently
            skipped += 1

    logger.info(
        f"LSH index: {inserted:,} candidates inserted, "
        f"{skipped:,} skipped (empty name or duplicate)"
    )
    return lsh, minhash_cache


def generate_ngram_lsh_pairs(
    source1: pd.DataFrame,
    lsh: "MinHashLSH",
    num_perm: int = 64,
    ngram_n: int = 3,
) -> pd.DataFrame:
    """
    Strategy: char_ngram_lsh
    Query LSH for each S1 entity and return approximate near-duplicate pairs.

    Returns DataFrame: source1_entity_id, candidate_entity_id
    """
    if not _HAS_DATASKETCH or lsh is None:
        logger.warning("N-gram LSH blocking skipped.")
        return pd.DataFrame(columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])

    records = []
    for row in source1.itertuples(index=False):
        s1_id = row.entity_id
        name = row.business_name_normalized

        if not name:
            continue

        ngrams = _char_ngrams(name, ngram_n)
        if not ngrams:
            continue

        m = _make_minhash(ngrams, num_perm)
        try:
            neighbors = lsh.query(m)
        except Exception as e:
            logger.debug(f"LSH query error for {s1_id}: {e}")
            continue

        for cand_id in neighbors:
            records.append((s1_id, cand_id))

    if not records:
        return pd.DataFrame(columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])

    return pd.DataFrame(records, columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])
