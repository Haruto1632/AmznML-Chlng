"""
blocking_eval.py — Evaluation utilities for blocking recall and efficiency.

Owner: Member A  |  Branch: feature/blocking
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from ..shared.data_loader import ground_truth_to_dict
from ..shared.schemas import (
    COL_SOURCE1_ID,
    COL_CANDIDATE_ID,
    COL_BLOCKING_REASONS,
    COL_MATCHED_IDS,
)

logger = logging.getLogger(__name__)


def evaluate_blocking(
    candidate_pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
    n_source1_total: int = None,
    n_source23_total: int = None,
) -> Dict:
    """
    Evaluate blocking quality against ground truth.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        Output of consolidate_candidates() or generate_candidates().
        Columns: source1_entity_id, candidate_entity_id [, blocking_reasons]
    ground_truth : pd.DataFrame
        Train ground truth. Columns: source1_entity_id, matched_entity_ids
    n_source1_total : int, optional
        Total number of S1 entities (for reduction ratio denominator).
    n_source23_total : int, optional
        Total number of S2+S3 entities (for reduction ratio denominator).

    Returns
    -------
    dict with keys:
        blocking_recall          -- fraction of true match pairs in candidate set
        total_candidates         -- total candidate pairs generated
        total_true_pairs         -- total true match pairs in ground truth
        true_pairs_recalled      -- true match pairs found in candidates
        avg_candidates_per_s1    -- mean candidates per S1 entity
        median_candidates_per_s1 -- median
        p95_candidates_per_s1    -- 95th percentile
        max_candidates_per_s1
        reduction_ratio          -- 1 - (candidates / brute_force_pairs)
        per_strategy_recall      -- dict: strategy → fraction of true matches it alone captures
        missed_pairs_sample      -- list of (s1_id, true_id) pairs NOT in candidates (sample)
    """
    gt_dict = ground_truth_to_dict(ground_truth)
    if not gt_dict:
        raise ValueError("Blocking evaluation requires non-empty ground truth")
    if not candidate_pairs[COL_SOURCE1_ID].isin(gt_dict).all():
        raise ValueError("Candidate Source 1 IDs must belong to the evaluation cohort")
    n_source1_total = len(gt_dict) if n_source1_total is None else n_source1_total
    if n_source1_total < len(gt_dict):
        raise ValueError("n_source1_total cannot be smaller than the ground truth cohort")
    unique_pairs = candidate_pairs.drop_duplicates([COL_SOURCE1_ID, COL_CANDIDATE_ID])

    # ── Build candidate lookup: s1_id → set(cand_ids) ────────────────────
    cand_lookup: Dict[str, set] = {}
    for row in candidate_pairs.itertuples(index=False):
        s1 = row.source1_entity_id
        cid = row.candidate_entity_id
        if s1 not in cand_lookup:
            cand_lookup[s1] = set()
        cand_lookup[s1].add(cid)

    # ── Compute recall ────────────────────────────────────────────────────
    total_true = 0
    recalled = 0
    missed_pairs: List[tuple] = []

    for s1_id, true_ids in gt_dict.items():
        cands = cand_lookup.get(s1_id, set())
        for tid in true_ids:
            total_true += 1
            if tid in cands:
                recalled += 1
            elif len(missed_pairs) < 200:  # sample only
                missed_pairs.append((s1_id, tid))

    blocking_recall = recalled / total_true if total_true > 0 else 0.0

    # ── Candidate count statistics ────────────────────────────────────────
    counts_per_s1 = unique_pairs.groupby(COL_SOURCE1_ID).size().reindex(gt_dict, fill_value=0)
    if n_source1_total > len(counts_per_s1):
        counts_per_s1 = pd.concat([counts_per_s1, pd.Series(0, index=range(n_source1_total - len(counts_per_s1)))])
    avg_cands = float(counts_per_s1.mean()) if len(counts_per_s1) > 0 else 0.0
    median_cands = float(counts_per_s1.median()) if len(counts_per_s1) > 0 else 0.0
    p95_cands = float(np.percentile(counts_per_s1.values, 95)) if len(counts_per_s1) > 0 else 0.0
    max_cands = int(counts_per_s1.max()) if len(counts_per_s1) > 0 else 0

    # ── Reduction ratio ───────────────────────────────────────────────────
    total_candidates = len(unique_pairs)
    if n_source1_total and n_source23_total:
        brute_force = n_source1_total * n_source23_total
        reduction_ratio = 1.0 - (total_candidates / brute_force)
    else:
        reduction_ratio = float("nan")

    # ── Per-strategy recall ───────────────────────────────────────────────
    per_strategy_recall: Dict[str, float] = {}
    if COL_BLOCKING_REASONS in candidate_pairs.columns and total_true > 0:
        # Build strategy → set of candidate_ids it covers
        strategy_cand_lookup: Dict[str, Dict[str, set]] = {}

        for row in candidate_pairs.itertuples(index=False):
            reasons = row.blocking_reasons
            if reasons is None:
                continue
            s1 = row.source1_entity_id
            cid = row.candidate_entity_id
            for strat in reasons:
                if strat not in strategy_cand_lookup:
                    strategy_cand_lookup[strat] = {}
                if s1 not in strategy_cand_lookup[strat]:
                    strategy_cand_lookup[strat][s1] = set()
                strategy_cand_lookup[strat][s1].add(cid)

        for strat, s_lookup in strategy_cand_lookup.items():
            s_recalled = sum(
                1
                for s1_id, true_ids in gt_dict.items()
                for tid in true_ids
                if tid in s_lookup.get(s1_id, set())
            )
            per_strategy_recall[strat] = s_recalled / total_true

    result = {
        "blocking_recall": blocking_recall,
        "total_candidates": total_candidates,
        "total_true_pairs": total_true,
        "true_pairs_recalled": recalled,
        "avg_candidates_per_s1": avg_cands,
        "median_candidates_per_s1": median_cands,
        "p95_candidates_per_s1": p95_cands,
        "max_candidates_per_s1": max_cands,
        "reduction_ratio": reduction_ratio,
        "per_strategy_recall": per_strategy_recall,
        "missed_pairs_sample": missed_pairs[:20],
    }

    return result


def print_blocking_report(metrics: Dict) -> None:
    """Pretty-print blocking evaluation results."""
    print("\n" + "=" * 60)
    print("BLOCKING EVALUATION REPORT")
    print("=" * 60)
    print(f"  Blocking recall        : {metrics['blocking_recall']:.4f}")
    print(f"  True pairs recalled    : {metrics['true_pairs_recalled']:,} / {metrics['total_true_pairs']:,}")
    print(f"  Total candidates       : {metrics['total_candidates']:,}")
    print(f"  Avg candidates per S1  : {metrics['avg_candidates_per_s1']:.1f}")
    print(f"  Median candidates/S1   : {metrics['median_candidates_per_s1']:.1f}")
    print(f"  P95 candidates/S1      : {metrics['p95_candidates_per_s1']:.1f}")
    print(f"  Max candidates/S1      : {metrics['max_candidates_per_s1']}")
    rr = metrics["reduction_ratio"]
    if not (isinstance(rr, float) and rr != rr):  # not nan
        print(f"  Reduction ratio        : {rr:.6f}")
    print("\n  Per-strategy recall:")
    for strat, recall in sorted(metrics["per_strategy_recall"].items()):
        print(f"    {strat:<35s}: {recall:.4f}")
    print("=" * 60)
