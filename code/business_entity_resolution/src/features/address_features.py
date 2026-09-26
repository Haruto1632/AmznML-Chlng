"""
address_features.py — Address similarity features for pairwise matching.

Owner: Member B  |  Branch: feature/matching-model

TODO (Member B): Implement compute_address_features().
See TEAM_TASKS.md Task B-1 for the full feature list.
"""

from __future__ import annotations

from typing import Dict

from ..shared.schemas import (
    COL_ADDRESS_NORMALIZED,
    COL_ADDRESS_NUMBERS,
    ADDRESS_FEATURE_COLUMNS,
)


def compute_address_features(rec_a: dict, rec_b: dict) -> Dict[str, float]:
    """
    Compute all address similarity features between two records.

    Parameters
    ----------
    rec_a : dict  — one record (S1 entity)
    rec_b : dict  — another record (S2/S3 candidate)

    Returns
    -------
    dict  — {feature_name: float_value} for all ADDRESS_FEATURE_COLUMNS
    All values numeric. Handle missing/empty addresses gracefully.
    """
    # TODO: implement
    raise NotImplementedError("compute_address_features() not yet implemented. See Task B-1.")
