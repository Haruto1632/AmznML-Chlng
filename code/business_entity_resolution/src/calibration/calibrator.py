"""
calibrator.py — Decision threshold calibration for F_0.5.

Owner: Member C  |  Branch: feature/evaluation-pipeline

Threshold selection uses the complete supplied validation cohort.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import math

import pandas as pd

from ..evaluation.evaluator import compute_f05_per_entity, compute_macro_f05
from ..shared.schemas import COL_SOURCE1_ID, COL_CANDIDATE_ID, COL_MATCH_PROB


def apply_threshold(
    predictions_df: pd.DataFrame,
    threshold: float,
) -> Dict[str, List[str]]:
    """
    Convert match probability DataFrame to a predictions dict using a threshold.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, match_probability
    threshold : float
        Pairs with match_probability >= threshold are included as matches.

    Returns
    -------
    dict
        {source1_entity_id: [matched_entity_ids, ...]}
        Entities with no pair above threshold → empty list (singleton prediction).
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between 0 and 1")
    if not predictions_df[COL_MATCH_PROB].between(0, 1).all():
        raise ValueError("Match probabilities must be finite and between 0 and 1")
    result = {key: [] for key in predictions_df[COL_SOURCE1_ID].unique()}
    accepted = predictions_df.loc[predictions_df[COL_MATCH_PROB] >= threshold]
    accepted = accepted.drop_duplicates([COL_SOURCE1_ID, COL_CANDIDATE_ID])
    result.update(accepted.groupby(COL_SOURCE1_ID, sort=False)[COL_CANDIDATE_ID].agg(list).to_dict())
    return result


def calibrate_threshold(
    predictions_df: pd.DataFrame,
    ground_truth: Dict[str, List[str]],
    all_s1_ids: List[str],
    thresholds: Optional[List[float]] = None,
) -> dict:
    """
    Sweep thresholds and select the one maximising macro F_0.5 on validation data.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, match_probability
        Contains only scored pairs. Zero-candidate entities belong in all_s1_ids.
    ground_truth : dict
        {source1_entity_id: [true_matched_ids]}
    all_s1_ids : list[str]
        All Source 1 entity IDs (to fill in entities with no candidates).
    thresholds : list[float], optional
        Thresholds to sweep. Default: 0.50 to 0.95 in steps of 0.05.

    Returns
    -------
    dict with:
        best_threshold   (float)
        best_f05         (float)
        threshold_curve  (pd.DataFrame: threshold, precision, recall, f05)
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in [x / 100 for x in range(50, 96, 5)]]
    if not thresholds:
        raise ValueError("At least one threshold is required")
    if not all_s1_ids or len(set(all_s1_ids)) != len(all_s1_ids):
        raise ValueError("Validation IDs must be non-empty and unique")
    if set(all_s1_ids) != set(ground_truth):
        raise ValueError("Ground truth must cover exactly the validation IDs")
    if not predictions_df[COL_SOURCE1_ID].isin(all_s1_ids).all():
        raise ValueError("Scored pairs contain IDs outside the validation cohort")

    rows = []
    for t in thresholds:
        preds = apply_threshold(predictions_df, t)
        # Ensure all S1 entities have an entry (singletons)
        for s1_id in all_s1_ids:
            if s1_id not in preds:
                preds[s1_id] = []
        per_entity = compute_f05_per_entity(preds, ground_truth)
        f05 = compute_macro_f05(per_entity)
        p = per_entity["precision"].mean()
        r = per_entity["recall"].mean()
        rows.append({"threshold": t, "precision": p, "recall": r, "f05": f05})

    curve_df = pd.DataFrame(rows)
    best_row = curve_df.sort_values(["f05", "threshold"], ascending=False).iloc[0]

    return {
        "best_threshold": float(best_row["threshold"]),
        "best_f05": float(best_row["f05"]),
        "threshold_curve": curve_df,
    }
