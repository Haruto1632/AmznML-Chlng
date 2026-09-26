"""
feature_builder.py — Pairwise feature engineering entry point.

Owner: Member B  |  Branch: feature/matching-model

TODO (Member B): Implement build_pair_features() using the sub-modules
name_features.py, address_features.py, country_features.py,
structural_features.py. See TEAM_TASKS.md Task B-1.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from ..shared.schemas import (
    COL_SOURCE1_ID,
    COL_CANDIDATE_ID,
    ALL_FEATURE_COLUMNS,
)


def build_pair_features(
    candidate_pairs: pd.DataFrame,
    records: Dict[str, dict],
    config: dict,
) -> pd.DataFrame:
    """
    Build a feature matrix for all candidate pairs.

    This is the PRIMARY interface used by the pipeline and scorer.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        Output of consolidate_candidates().
        Columns: source1_entity_id, candidate_entity_id, blocking_reasons
    records : dict
        {entity_id: {column: value, ...}} lookup for all records
        (both source1 and candidates). Build with data_loader.records_to_dict().
    config : dict
        Pipeline config. Uses config["features"].

    Returns
    -------
    pd.DataFrame
        Columns:
          source1_entity_id   (str)
          candidate_entity_id (str)
          <feature columns>   (float) — see schemas.ALL_FEATURE_COLUMNS
        One row per candidate pair. No NaN values (fill missing with 0.0).
    """
    # TODO: implement
    #   for each pair:
    #     s1_rec = records[source1_entity_id]
    #     cand_rec = records[candidate_entity_id]
    #     name_feats = name_features.compute(s1_rec, cand_rec)
    #     addr_feats = address_features.compute(s1_rec, cand_rec)
    #     ...
    #     combine into one row
    raise NotImplementedError(
        "build_pair_features() not yet implemented. See TEAM_TASKS.md Task B-1."
    )
