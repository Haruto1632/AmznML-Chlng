"""
blocker.py — Multi-strategy blocking orchestrator.

Owner: Member A  |  Branch: feature/blocking

TODO (Member A): Implement generate_candidates() by calling each strategy
and unioning the results. See TEAM_TASKS.md Task A-3.
"""

from __future__ import annotations

import pandas as pd

from ..shared.schemas import (
    COL_SOURCE1_ID,
    COL_CANDIDATE_ID,
    COL_BLOCKING_REASONS,
)


def generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Generate candidate pairs using multiple blocking strategies.

    This is the PRIMARY interface used by the pipeline.
    All strategies are run and their results are unioned.

    Parameters
    ----------
    source1 : pd.DataFrame
        Normalized Source 1 records.
    source2 : pd.DataFrame
        Normalized Source 2 records.
    source3 : pd.DataFrame
        Normalized Source 3 records.
    config : dict
        Pipeline config (see src/pipeline/config.py).

    Returns
    -------
    pd.DataFrame
        Columns:
          source1_entity_id   (str)
          candidate_entity_id (str)
          blocking_reasons    (list[str])
        One row per unique (source1_entity_id, candidate_entity_id) pair.
        Every S1 entity appears at least once (with empty candidates if no
        strategy produced any pair for it — singletons).
    """
    # TODO: implement
    #
    # Pseudocode:
    #   all_pairs = []
    #   if config["blocking"]["strategy_exact_token"]:
    #       all_pairs.append(run_exact_token(source1, source2, source3))
    #   if config["blocking"]["strategy_token_overlap"]:
    #       all_pairs.append(run_token_overlap(...))
    #   ... etc.
    #   return consolidate(all_pairs)
    raise NotImplementedError(
        "generate_candidates() not yet implemented. See TEAM_TASKS.md Task A-3."
    )
