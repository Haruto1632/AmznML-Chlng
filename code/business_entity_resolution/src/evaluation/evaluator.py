"""
evaluator.py — F_0.5 evaluation framework.

Owner: Member C  |  Branch: feature/evaluation-pipeline

Macro metrics include the complete supplied Source 1 evaluation cohort.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ..shared.schemas import COL_SOURCE1_ID


def _f05(precision: float, recall: float) -> float:
    """
    Compute F_0.5 for a single precision/recall pair.

    F_0.5 = (1 + 0.5^2) * P * R / (0.5^2 * P + R)
           = 1.25 * P * R / (0.25 * P + R)

    Returns 0.0 if both precision and recall are 0.
    Returns 1.0 for a perfect singleton prediction (P=1, R=1 trivially).
    """
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def compute_f05_per_entity(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Compute F_0.5 for every Source 1 entity.

    Parameters
    ----------
    predictions : dict
        {source1_entity_id: [matched_ids, ...]}  — may be empty list
    ground_truth : dict
        {source1_entity_id: [true_ids, ...]}  — may be empty list

    Returns
    -------
    pd.DataFrame
        Columns:
          source1_entity_id, precision, recall, f05, is_singleton, is_correct_singleton
    """
    if not ground_truth:
        raise ValueError("Evaluation requires non-empty ground truth")
    if predictions.keys() - ground_truth.keys():
        raise ValueError("Predictions contain Source 1 IDs outside the evaluation cohort")
    rows = []
    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, [])
        true_set = set(true_ids)
        pred_set = set(pred_ids)
        is_singleton = len(true_set) == 0

        if is_singleton:
            # Predict empty → 1.0 | predict anything → 0.0
            is_correct_singleton = len(pred_set) == 0
            f05 = 1.0 if is_correct_singleton else 0.0
            precision = 1.0 if is_correct_singleton else 0.0
            recall = 1.0
        else:
            tp = len(true_set & pred_set)
            precision = tp / len(pred_set) if pred_set else 0.0
            recall = tp / len(true_set)
            f05 = _f05(precision, recall)
            is_correct_singleton = False

        rows.append({
            COL_SOURCE1_ID: s1_id,
            "precision": precision,
            "recall": recall,
            "f05": f05,
            "is_singleton": is_singleton,
            "is_correct_singleton": is_correct_singleton,
        })
    return pd.DataFrame(rows)


def compute_macro_f05(per_entity_df: pd.DataFrame) -> float:
    """Macro-average F_0.5 across all Source 1 entities."""
    return per_entity_df["f05"].mean()


def full_evaluation_report(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    candidate_pairs: pd.DataFrame = None,
) -> dict:
    """
    Generate a full evaluation report.

    Parameters
    ----------
    predictions : dict
        {source1_entity_id: [matched_ids]}
    ground_truth : dict
        {source1_entity_id: [true_ids]}
    candidate_pairs : pd.DataFrame, optional
        For computing blocking recall and candidate statistics.

    Returns
    -------
    dict with full metrics.
    """
    per_entity = compute_f05_per_entity(predictions, ground_truth)

    total = len(per_entity)
    singletons = per_entity["is_singleton"].sum()

    report = {
        "macro_f05": compute_macro_f05(per_entity),
        "macro_precision": per_entity["precision"].mean(),
        "macro_recall": per_entity["recall"].mean(),
        "total_entities": total,
        "singleton_count": int(singletons),
        "singleton_accuracy": (
            per_entity.loc[per_entity["is_singleton"], "is_correct_singleton"].mean()
            if singletons > 0 else None
        ),
        "non_singleton_f05": (
            per_entity.loc[~per_entity["is_singleton"], "f05"].mean()
            if (total - singletons) > 0 else None
        ),
    }

    # Blocking stats (if candidate_pairs provided)
    if candidate_pairs is not None:
        from ..candidate_generation.blocking_eval import evaluate_blocking
        gt_df = pd.DataFrame({
            COL_SOURCE1_ID: list(ground_truth),
            "matched_entity_ids": [",".join(ids) for ids in ground_truth.values()],
        })
        report.update(evaluate_blocking(candidate_pairs, gt_df))

    return report


def print_report(report: dict) -> None:
    """Pretty-print an evaluation report dict."""
    print("=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    for k, v in report.items():
        if v is None:
            print(f"  {k:35s}: N/A")
        elif isinstance(v, float):
            print(f"  {k:35s}: {v:.4f}")
        else:
            print(f"  {k:35s}: {v}")
    print("=" * 50)
