"""
blocking_eval.py — Evaluation utilities for blocking recall and efficiency.

Owner: Member A  |  Branch: feature/blocking

TODO (Member A): Implement evaluate_blocking() and log results to
experiments/blocking_v1.md. See TEAM_TASKS.md Task A-5.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from ..shared.data_loader import ground_truth_to_dict
from ..shared.schemas import COL_SOURCE1_ID, COL_CANDIDATE_ID, COL_BLOCKING_REASONS, COL_MATCHED_IDS


def evaluate_blocking(
    candidate_pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> Dict[str, float]:
    """
    Evaluate blocking quality against ground truth.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        Output of consolidate_candidates().
        Columns: source1_entity_id, candidate_entity_id, blocking_reasons
    ground_truth : pd.DataFrame
        Train ground truth. Columns: source1_entity_id, matched_entity_ids

    Returns
    -------
    dict with keys:
        blocking_recall          -- fraction of true match pairs surviving blocking
        total_candidates         -- total candidate pairs
        total_true_pairs         -- total true match pairs in ground truth
        avg_candidates_per_s1
        median_candidates_per_s1
        p95_candidates_per_s1
        reduction_ratio          -- 1 - (candidates / brute_force_pairs)
        per_strategy_recall      -- dict: strategy -> fraction of true matches it alone captures
    """
    # TODO: implement
    raise NotImplementedError(
        "evaluate_blocking() not yet implemented. See TEAM_TASKS.md Task A-5."
    )
