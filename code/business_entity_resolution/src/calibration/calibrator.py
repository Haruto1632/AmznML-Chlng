"""
calibrator.py — Decision threshold calibration for F_0.5.

Owner: Member C  |  Branch: feature/evaluation-pipeline

TODO (Member C): Implement calibrate_threshold().
See TEAM_TASKS.md Task C-3.
"""

from __future__ import annotations

from typing import Dict, List, Optional

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
    result: Dict[str, List[str]] = {}
    for _, row in predictions_df.iterrows():
        s1_id = row[COL_SOURCE1_ID]
        if s1_id not in result:
            result[s1_id] = []
        if row[COL_MATCH_PROB] >= threshold:
            result[s1_id].append(row[COL_CANDIDATE_ID])
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
        Must cover ALL Source 1 entities (even those with no candidates).
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
    best_row = curve_df.loc[curve_df["f05"].idxmax()]

    return {
        "best_threshold": float(best_row["threshold"]),
        "best_f05": float(best_row["f05"]),
        "threshold_curve": curve_df,
    }
