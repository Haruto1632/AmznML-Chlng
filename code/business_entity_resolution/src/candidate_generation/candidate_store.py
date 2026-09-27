"""
candidate_store.py — Candidate consolidation, deduplication, and limiting.

Owner: Member A  |  Branch: feature/blocking
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from ..shared.schemas import COL_SOURCE1_ID, COL_CANDIDATE_ID, COL_BLOCKING_REASONS

logger = logging.getLogger(__name__)


def consolidate_candidates(
    raw_pairs: pd.DataFrame,
    source1_ids: List[str],
    config: dict,
) -> pd.DataFrame:
    """
    Deduplicate, merge blocking reasons, apply candidate-count limits.

    Parameters
    ----------
    raw_pairs : pd.DataFrame
        Output of blocker.generate_candidates().
        Columns: source1_entity_id, candidate_entity_id, blocking_reasons (list)
    source1_ids : list[str]
        All Source 1 entity IDs — ensures every S1 has an entry.
    config : dict
        Pipeline config. Uses:
            config["candidate_generation"]["max_candidates_per_s1"] (int or None)
            config["candidate_generation"]["prefer_high_evidence"] (bool)

    Returns
    -------
    pd.DataFrame
        Columns:
          source1_entity_id   (str)
          candidate_entity_id (str)
          blocking_reasons    (list[str])
          num_strategies      (int)
        Every S1 entity appears at least once (may have 0 candidate rows
        — use write_candidate_pairs to handle empty candidates correctly).
        Pairs sorted within each S1 group: most-evidenced pairs first.
    """
    cand_cfg = config.get("candidate_generation", {})
    max_cands = cand_cfg.get("max_candidates_per_s1", None)
    prefer_high_evidence = cand_cfg.get("prefer_high_evidence", True)

    if raw_pairs is None or len(raw_pairs) == 0:
        logger.warning("No raw pairs to consolidate.")
        # Return empty DataFrame with correct schema
        return pd.DataFrame(
            columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID, COL_BLOCKING_REASONS, "num_strategies"]
        )

    # ── Deduplicate: (s1_id, cand_id) pairs should be unique ─────────────
    # blocking_reasons is a list; we need to union them per unique pair.
    # groupby + agg is the vectorized approach.
    df = raw_pairs.copy()

    # Ensure blocking_reasons is always a list
    if COL_BLOCKING_REASONS not in df.columns:
        df[COL_BLOCKING_REASONS] = [[] for _ in range(len(df))]

    # Check if already deduplicated (blocker.py does this via defaultdict)
    is_unique = ~df.duplicated(subset=[COL_SOURCE1_ID, COL_CANDIDATE_ID]).any()

    if not is_unique:
        logger.info("Deduplicating pairs...")
        # Aggregate: union of blocking_reasons lists per pair
        agg = (
            df.groupby([COL_SOURCE1_ID, COL_CANDIDATE_ID], sort=False)
            .agg({COL_BLOCKING_REASONS: lambda x: list(set(r for lst in x for r in lst))})
            .reset_index()
        )
        df = agg

    # Add num_strategies column
    df["num_strategies"] = df[COL_BLOCKING_REASONS].apply(len)

    # ── Apply candidate limit ─────────────────────────────────────────────
    if max_cands is not None:
        logger.info("Applying candidate limit: max %d per S1...", max_cands)
        if prefer_high_evidence:
            # Sort by evidence count (descending) then take top N
            df = df.sort_values(
                [COL_SOURCE1_ID, "num_strategies"],
                ascending=[True, False]
            )
        df = (
            df.groupby(COL_SOURCE1_ID, sort=False)
            .head(max_cands)
            .reset_index(drop=True)
        )

    logger.info(
        "Consolidation complete: %d pairs across %d S1 entities",
        len(df),
        df[COL_SOURCE1_ID].nunique() if len(df) > 0 else 0,
    )

    return df
