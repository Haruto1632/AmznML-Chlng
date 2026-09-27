"""
token_blocking.py — Token inverted-index blocking strategies.

Owner: Member A  |  Branch: feature/blocking

Implements two strategies:
  1. exact_country_name_token: same country + ≥1 shared name token
  2. name_token_overlap: ≥1 shared name token (country-agnostic)

SCALABILITY DESIGN:
  - Index built in two passes over candidates (no explode+groupby memory spike)
  - Two-tier stop-token filtering:
    (a) Global DF threshold: skip tokens in > max_df_frac of ALL records
    (b) Per-(country, token) bucket cap: skip token lookups where bucket exceeds
        max_bucket_size (avoids blowing up on "services" in US)
  - Query: itertuples loop over S1 × dict lookup (no Python inner loop over candidates)
  - Returns list of (s1_id, cand_id) tuples → DataFrame conversion by caller
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, Optional, Set, Tuple

import pandas as pd

from ...shared.schemas import (
    COL_ENTITY_ID,
    COL_NAME_TOKENS,
    COL_COUNTRY_NORMALIZED,
    COL_SOURCE1_ID,
    COL_CANDIDATE_ID,
    STRATEGY_EXACT_TOKEN,
    STRATEGY_TOKEN_OVERLAP,
)

logger = logging.getLogger(__name__)


def _build_token_index(
    candidates: pd.DataFrame,
    max_df_frac: float = 0.001,
    max_bucket_size: int = 5000,
) -> Tuple[Dict, Dict, Set[str]]:
    """
    Build inverted indices from normalized candidate records.

    Two-pass approach using defaultdict. No large intermediate DataFrames.

    Parameters
    ----------
    candidates : pd.DataFrame
        Combined S2+S3 normalized DataFrame with columns:
        entity_id, business_name_tokens, country_normalized
    max_df_frac : float
        Tokens appearing in more than this fraction of ALL records are
        treated as stop-tokens and excluded from blocking.
        Default 0.001 = 0.1% of records (~10K records in 10M pool).
        Increase to 0.01 for less aggressive filtering (may slow queries).
    max_bucket_size : int
        Country+token index entries with more than this many candidates
        are treated as too ambiguous and will be skipped at query time.
        This prevents O(N) enumeration of massive buckets.
        Default 5000.

    Returns
    -------
    country_token_index : dict
        (country_normalized, token) → set(entity_ids)
        Only entries with ≤ max_bucket_size candidates.
    token_index : dict
        token → set(entity_ids)
        Only entries with ≤ max_bucket_size candidates.
    stop_tokens : set
        Tokens excluded due to high global document frequency.
    """
    n_candidates = len(candidates)
    max_df = max(1, int(n_candidates * max_df_frac))

    # --- Pass 1: count global document frequency per token ---
    token_doc_freq: Counter = Counter()
    for row in candidates.itertuples(index=False):
        tokens = row.business_name_tokens  # frozenset
        if not tokens:
            continue
        for tok in tokens:
            if len(tok) >= 2:
                token_doc_freq[tok] += 1

    stop_tokens: Set[str] = {tok for tok, cnt in token_doc_freq.items() if cnt > max_df}
    logger.info(f"Stop-tokens (df > {max_df}, {max_df_frac*100:.2f}% of {n_candidates:,}): "
                f"{len(stop_tokens)} tokens excluded")
    del token_doc_freq  # free memory

    # --- Pass 2: build inverted indices ---
    country_token_idx: Dict = defaultdict(set)
    token_idx: Dict = defaultdict(set)

    for row in candidates.itertuples(index=False):
        eid = row.entity_id
        country = row.country_normalized
        tokens = row.business_name_tokens  # frozenset
        if not tokens:
            continue
        for tok in tokens:
            if len(tok) < 2 or tok in stop_tokens:
                continue
            token_idx[tok].add(eid)
            country_token_idx[(country, tok)].add(eid)

    # Filter out oversized buckets (too ambiguous for blocking)
    oversized_ct = sum(1 for v in country_token_idx.values() if len(v) > max_bucket_size)
    oversized_t = sum(1 for v in token_idx.values() if len(v) > max_bucket_size)

    country_token_index = {k: v for k, v in country_token_idx.items()
                           if len(v) <= max_bucket_size}
    token_index = {k: v for k, v in token_idx.items()
                   if len(v) <= max_bucket_size}

    logger.info(
        f"Token index: {len(token_index):,} tokens "
        f"({oversized_t} oversized buckets filtered), "
        f"covering {n_candidates:,} candidates"
    )
    logger.info(
        f"Country+token index: {len(country_token_index):,} keys "
        f"({oversized_ct} oversized filtered)"
    )
    return country_token_index, token_index, stop_tokens


def generate_exact_country_token_pairs(
    source1: pd.DataFrame,
    candidates: pd.DataFrame,
    country_token_index: Dict,
    stop_tokens: Set[str],
    max_cands_per_s1: Optional[int] = None,
) -> pd.DataFrame:
    """
    Strategy: exact_country_name_token
    Generates pairs where S1 and candidate share the same country AND
    at least one non-stop name token whose bucket is within size limit.

    Parameters
    ----------
    max_cands_per_s1 : int, optional
        Hard cap on candidates generated per S1 entity from this strategy.
        When set, we stop after collecting this many candidates.

    Returns DataFrame: source1_entity_id, candidate_entity_id
    """
    records = []
    for row in source1.itertuples(index=False):
        s1_id = row.entity_id
        country = row.country_normalized
        tokens = row.business_name_tokens  # frozenset

        seen: Set[str] = set()
        for tok in tokens:
            if tok in stop_tokens:
                continue
            key = (country, tok)
            for cand_id in country_token_index.get(key, ()):
                if cand_id not in seen:
                    seen.add(cand_id)
                    records.append((s1_id, cand_id))
                    if max_cands_per_s1 and len(seen) >= max_cands_per_s1:
                        break
            if max_cands_per_s1 and len(seen) >= max_cands_per_s1:
                break

    if not records:
        return pd.DataFrame(columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])

    return pd.DataFrame(records, columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])


def generate_token_overlap_pairs(
    source1: pd.DataFrame,
    candidates: pd.DataFrame,
    token_index: Dict,
    stop_tokens: Set[str],
    max_cands_per_s1: Optional[int] = None,
) -> pd.DataFrame:
    """
    Strategy: name_token_overlap
    Generates pairs where S1 and candidate share at least one non-stop name token
    (without requiring same country — catches cross-country duplicates).

    Returns DataFrame: source1_entity_id, candidate_entity_id
    """
    records = []
    for row in source1.itertuples(index=False):
        s1_id = row.entity_id
        tokens = row.business_name_tokens  # frozenset

        seen: Set[str] = set()
        for tok in tokens:
            if tok in stop_tokens:
                continue
            for cand_id in token_index.get(tok, ()):
                if cand_id not in seen:
                    seen.add(cand_id)
                    records.append((s1_id, cand_id))
                    if max_cands_per_s1 and len(seen) >= max_cands_per_s1:
                        break
            if max_cands_per_s1 and len(seen) >= max_cands_per_s1:
                break

    if not records:
        return pd.DataFrame(columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])

    return pd.DataFrame(records, columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])
