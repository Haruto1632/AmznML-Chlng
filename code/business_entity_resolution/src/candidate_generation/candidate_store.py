"""
candidate_store.py — Candidate consolidation, deduplication, and limiting.

Owner: Member A  |  Branch: feature/blocking

TODO (Member A): Implement consolidate_candidates() and write the final
candidate_pairs.tsv. See TEAM_TASKS.md Task A-4.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from ..shared.schemas import COL_SOURCE1_ID, COL_CANDIDATE_ID, COL_BLOCKING_REASONS


def consolidate_candidates(
    raw_pairs: pd.DataFrame,
    source1_ids: List[str],
    config: dict,
) -> pd.DataFrame:
    """
    Deduplicate, merge blocking reasons, and apply candidate-count limits.

    Parameters
    ----------
    raw_pairs : pd.DataFrame
        Output of blocker.generate_candidates().
        Columns: source1_entity_id, candidate_entity_id, blocking_reasons
    source1_ids : list[str]
        All Source 1 entity IDs — guarantees every S1 entity has a row.
    config : dict
        Pipeline config. Uses config["candidate_generation"]["max_candidates_per_s1"].

    Returns
    -------
    pd.DataFrame
        Columns:
          source1_entity_id   (str)
          candidate_entity_id (str)
          blocking_reasons    (list[str])  — union of all strategies that produced this pair
        Every S1 entity appears at least once.
        Pairs are sorted by evidence strength (more strategies = higher priority).
    """
    # TODO: implement
    raise NotImplementedError(
        "consolidate_candidates() not yet implemented. See TEAM_TASKS.md Task A-4."
    )
