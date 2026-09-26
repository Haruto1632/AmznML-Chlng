"""
name_features.py — Name similarity features for pairwise matching.

Owner: Member B  |  Branch: feature/matching-model

TODO (Member B): Implement compute_name_features().
See TEAM_TASKS.md Task B-1 for the full feature list.
"""

from __future__ import annotations

from typing import Dict

from ..shared.schemas import (
    COL_NAME_NORMALIZED,
    COL_NAME_TOKENS,
    COL_NAME_TOKEN_SORTED,
    NAME_FEATURE_COLUMNS,
)


def compute_name_features(rec_a: dict, rec_b: dict) -> Dict[str, float]:
    """
    Compute all name similarity features between two records.

    Parameters
    ----------
    rec_a : dict  — one record (e.g., S1 entity), must have normalized columns
    rec_b : dict  — another record (e.g., S2/S3 candidate)

    Returns
    -------
    dict  — {feature_name: float_value} for all NAME_FEATURE_COLUMNS
    All values must be numeric (float). No NaN, no None.
    """
    # TODO: implement
    # Suggested libraries: rapidfuzz for Levenshtein/Jaro-Winkler/token ratios
    raise NotImplementedError("compute_name_features() not yet implemented. See Task B-1.")
