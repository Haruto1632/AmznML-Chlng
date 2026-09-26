"""
exact_token.py — Exact country + name token blocking.

Strategy: pair (S1, S2/S3) if they share the same country AND at least
one normalised name token.

Owner: Member A  |  Branch: feature/blocking
"""

from __future__ import annotations

import pandas as pd

from ...shared.schemas import (
    COL_ENTITY_ID,
    COL_NAME_TOKENS,
    COL_COUNTRY_NORMALIZED,
    COL_SOURCE1_ID,
    COL_CANDIDATE_ID,
    STRATEGY_EXACT_TOKEN,
)


def run(
    source1: pd.DataFrame,
    candidates: pd.DataFrame,  # union of S2 and S3 (already normalized)
) -> pd.DataFrame:
    """
    Returns DataFrame with columns (source1_entity_id, candidate_entity_id).
    """
    # TODO: implement
    # Approach:
    #   1. Build inverted index: country → {token → [entity_ids]}
    #   2. For each S1 entity, look up its country + tokens in S2/S3 index
    #   3. Return all matching pairs
    raise NotImplementedError("exact_token.run() not yet implemented.")
