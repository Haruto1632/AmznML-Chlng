"""
ngram_lsh.py — Character n-gram MinHash LSH blocking.

Strategy: use datasketch MinHashLSH to find approximate near-duplicates
based on character n-gram sets of normalised business names.

Owner: Member A  |  Branch: feature/blocking

Requires: datasketch (see requirements.txt)
"""

from __future__ import annotations

import pandas as pd

from ...shared.schemas import STRATEGY_NGRAM_LSH


def _char_ngrams(text: str, n: int = 3) -> set:
    """Extract character n-grams from a string."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def run(
    source1: pd.DataFrame,
    candidates: pd.DataFrame,
    ngram_n: int = 3,
    num_perm: int = 128,
    threshold: float = 0.25,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns (source1_entity_id, candidate_entity_id).

    Uses datasketch.MinHashLSH for approximate Jaccard similarity search
    over character n-grams.
    """
    # TODO: implement using datasketch
    # Approach:
    #   1. Build MinHash for each S2/S3 entity name
    #   2. Insert all S2/S3 MinHashes into an LSH index
    #   3. For each S1 entity, query the LSH index
    #   4. Return all pairs above threshold
    raise NotImplementedError("ngram_lsh.run() not yet implemented.")
