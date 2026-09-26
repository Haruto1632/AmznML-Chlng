"""
scorer.py — Wraps the model to produce a standardised probability DataFrame.

Owner: Member B  |  Branch: feature/matching-model

TODO (Member B): Implement predict_match_probabilities().
See TEAM_TASKS.md Task B-4.
"""

from __future__ import annotations

import pandas as pd

from ..shared.schemas import (
    COL_SOURCE1_ID,
    COL_CANDIDATE_ID,
    COL_MATCH_PROB,
    ALL_FEATURE_COLUMNS,
)


def predict_match_probabilities(
    feature_df: pd.DataFrame,
    model,
    config: dict,
) -> pd.DataFrame:
    """
    Run the model and return match probabilities.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Output of feature_builder.build_pair_features().
        Must contain all columns in ALL_FEATURE_COLUMNS.
    model
        Trained model with a predict_proba(X) method.
    config : dict
        Pipeline config.

    Returns
    -------
    pd.DataFrame
        Columns:
          source1_entity_id   (str)
          candidate_entity_id (str)
          match_probability   (float in [0, 1])
    """
    # TODO: implement
    raise NotImplementedError(
        "predict_match_probabilities() not yet implemented. See TEAM_TASKS.md Task B-4."
    )
