"""
address_token.py — Address token overlap blocking.

Strategy: pair (S1, S2/S3) if they share at least one numeric address token
AND at least one non-numeric address token.

Owner: Member A  |  Branch: feature/blocking
"""

from __future__ import annotations

import pandas as pd

from ...shared.schemas import STRATEGY_ADDRESS_TOKEN, COL_ADDRESS_NORMALIZED, COL_ADDRESS_NUMBERS


def run(
    source1: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns (source1_entity_id, candidate_entity_id).
    """
    # TODO: implement
    # Approach:
    #   1. For S2/S3: build inverted index on address number tokens
    #   2. For each S1 entity, find S2/S3 entities sharing a number token
    #   3. Among those, further filter by street token overlap
    raise NotImplementedError("address_token.run() not yet implemented.")
