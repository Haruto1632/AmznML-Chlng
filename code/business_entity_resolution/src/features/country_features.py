"""
country_features.py — Country comparison features.

Owner: Member B  |  Branch: feature/matching-model
"""

from __future__ import annotations

from typing import Dict

from ..shared.schemas import COL_COUNTRY_NORMALIZED, COUNTRY_FEATURE_COLUMNS


def compute_country_features(rec_a: dict, rec_b: dict) -> Dict[str, float]:
    """
    Compute country comparison features between two records.

    Returns
    -------
    dict with COUNTRY_FEATURE_COLUMNS keys.
    """
    a_country = rec_a.get(COL_COUNTRY_NORMALIZED, "")
    b_country = rec_b.get(COL_COUNTRY_NORMALIZED, "")

    exact = 1.0 if (a_country and b_country and a_country == b_country) else 0.0
    both_missing = 1.0 if (not a_country and not b_country) else 0.0

    return {
        "country_exact_normalized": exact,
        "country_both_missing": both_missing,
    }
