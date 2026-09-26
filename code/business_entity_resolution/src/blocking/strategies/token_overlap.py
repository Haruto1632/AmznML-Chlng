"""
token_overlap.py — Name token Jaccard overlap blocking.

Strategy: pair (S1, S2/S3) if Jaccard(name_tokens_A, name_tokens_B) >= threshold.

Owner: Member A  |  Branch: feature/blocking
"""

from __future__ import annotations

import pandas as pd

from ...shared.schemas import STRATEGY_TOKEN_OVERLAP


def run(
    source1: pd.DataFrame,
    candidates: pd.DataFrame,
    threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns (source1_entity_id, candidate_entity_id).
    
    Tip: Use inverted index on tokens for efficiency — do NOT do O(n*m) comparisons.
    Build: token → list of entity_ids, then for each S1 entity find all S2/S3 
    entities sharing at least one token, then compute Jaccard only on those.
    """
    # TODO: implement
    raise NotImplementedError("token_overlap.run() not yet implemented.")
