"""
structural_features.py — Blocking evidence / structural features.

Owner: Member B  |  Branch: feature/matching-model
"""

from __future__ import annotations

from typing import Dict, List

from ..shared.schemas import (
    COL_BLOCKING_REASONS,
    STRUCTURAL_FEATURE_COLUMNS,
    ALL_STRATEGIES,
)


def compute_structural_features(blocking_reasons: List[str]) -> Dict[str, float]:
    """
    Compute structural features from blocking evidence.

    Parameters
    ----------
    blocking_reasons : list[str]
        List of strategy names that produced this candidate pair.

    Returns
    -------
    dict with STRUCTURAL_FEATURE_COLUMNS keys.
    """
    reasons_set = set(blocking_reasons) if blocking_reasons else set()
    features = {
        "num_blocking_strategies": float(len(reasons_set)),
    }
    for strategy in ALL_STRATEGIES:
        features[f"strategy_{strategy}"] = 1.0 if strategy in reasons_set else 0.0
    return features
