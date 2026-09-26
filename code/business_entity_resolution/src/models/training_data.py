"""
training_data.py — Training pair construction from ground truth.

Owner: Member B  |  Branch: feature/matching-model

TODO (Member B): Implement build_training_pairs() with proper
positive/negative sampling including hard negatives.
See TEAM_TASKS.md Task B-2.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ..shared.schemas import COL_SOURCE1_ID, COL_MATCHED_IDS, COL_LABEL


def build_training_pairs(
    ground_truth: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    feature_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Construct labeled training pairs.

    Positive pairs: from ground_truth (true matches).
    Negative pairs: from candidate_pairs that are NOT in ground_truth.
      Mix of random negatives and hard negatives (see hard_negatives.py).

    Parameters
    ----------
    ground_truth : pd.DataFrame
        Columns: source1_entity_id, matched_entity_ids
    candidate_pairs : pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, blocking_reasons
    feature_df : pd.DataFrame
        Precomputed features for all candidate pairs.
    config : dict
        Uses config["model"]["negative_sample_ratio"] and
        config["model"]["hard_negative_fraction"].

    Returns
    -------
    pd.DataFrame
        feature_df columns + "label" column (1 = match, 0 = non-match).
    """
    # TODO: implement
    raise NotImplementedError(
        "build_training_pairs() not yet implemented. See TEAM_TASKS.md Task B-2."
    )
