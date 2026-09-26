"""
matcher.py — Matching model interface and baseline rule-based scorer.

Owner: Member B  |  Branch: feature/matching-model

TODO (Member B):
  1. Implement baseline_score() — simple weighted rule baseline
  2. Implement GBMMatcherModel in gbm_model.py
  3. Wire everything through predict_proba() here
See TEAM_TASKS.md Tasks B-2 through B-4.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from ..shared.schemas import COL_SOURCE1_ID, COL_CANDIDATE_ID, COL_MATCH_PROB


def baseline_score(feature_row: dict) -> float:
    """
    Simple rule-based baseline score. Implement this FIRST.

    Score = 0.5 * name_token_jaccard
          + 0.3 * addr_token_jaccard
          + 0.2 * country_exact_normalized

    Parameters
    ----------
    feature_row : dict
        One row of features from feature_builder.build_pair_features().

    Returns
    -------
    float in [0, 1]
    """
    # TODO: implement
    raise NotImplementedError(
        "baseline_score() not yet implemented. See TEAM_TASKS.md Task B-5."
    )


def predict_proba(
    feature_df: pd.DataFrame,
    model,
    config: dict,
) -> pd.DataFrame:
    """
    Apply a trained model to produce match probabilities.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Output of feature_builder.build_pair_features().
    model
        Trained model object (GBMMatcherModel or a callable with predict_proba).
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
        "predict_proba() not yet implemented. See TEAM_TASKS.md Task B-4."
    )
