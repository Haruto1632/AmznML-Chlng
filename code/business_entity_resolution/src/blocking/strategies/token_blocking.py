"""
token_blocking.py — Token inverted-index blocking strategies.

Owner: Member A  |  Branch: feature/blocking

Implements two strategies:
  1. exact_country_name_token: same country + ≥1 shared name token
  2. name_token_overlap: ≥1 shared name token (country-agnostic)

Both use a pre-built inverted index: token → set(candidate_ids).
The index is built once over S2+S3; S1 queries it.

Scalability design:
  - Index built in a single vectorized pass (explode + groupby)
  - IDF-based token cutoff: skip tokens appearing in > max_token_df_frac of records
  - Query via dict lookup — no nested loops over candidate pool
  - Returns set of (s1_id, cand_id) tuples → converted to DataFrame by caller
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Set, Tuple

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
    max_df_frac: float = 0.01,
) -> Tuple[Dict, Dict, Set[str]]:
    """
    Build inverted indices from normalized candidate records.

    Two-pass defaultdict approach — avoids creating a 30M-row intermediate
    DataFrame from explode+groupby, which was the main bottleneck.

    Pass 1: Count token document frequencies (for stop-token filtering).
    Pass 2: Build inverted index, skipping stop-tokens.

    Parameters
    ----------
    candidates : pd.DataFrame
        Combined S2+S3 normalized DataFrame with columns:
        entity_id, business_name_tokens, country_normalized
    max_df_frac : float
        Tokens appearing in more than this fraction of ALL records are
        treated as stop-tokens and excluded from blocking.
        Default 0.01 = 1% of records.

    Returns
    -------
    country_token_index : dict
        (country_normalized, token) → set(entity_ids)
    token_index : dict
        token → set(entity_ids)
    stop_tokens : set
        Tokens excluded due to high document frequency.
    """
    from collections import Counter
    n_candidates = len(candidates)
    max_df = max(1, int(n_candidates * max_df_frac))

    # --- Pass 1: count document frequency per token ---
    token_doc_freq: Counter = Counter()
    for row in candidates.itertuples(index=False):
        tokens = row.business_name_tokens  # frozenset
        if not tokens:
            continue
        for tok in tokens:
            if len(tok) >= 2:
                token_doc_freq[tok] += 1

    stop_tokens: Set[str] = {tok for tok, cnt in token_doc_freq.items() if cnt > max_df}
    logger.info(f"Stop-tokens (df > {max_df}): {len(stop_tokens)} tokens excluded")
    del token_doc_freq  # free memory

    # --- Pass 2: build inverted indices (skip stop-tokens) ---
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

    logger.info(
        f"Token index: {len(token_idx):,} tokens covering "
        f"{n_candidates:,} candidates"
    )
    return dict(country_token_idx), dict(token_idx), stop_tokens


def generate_exact_country_token_pairs(
    source1: pd.DataFrame,
    candidates: pd.DataFrame,
    country_token_index: Dict,
    stop_tokens: Set[str],
) -> pd.DataFrame:
    """
    Strategy: exact_country_name_token
    Generates pairs where S1 and candidate share the same country AND
    at least one non-stop name token.

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

    if not records:
        return pd.DataFrame(columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])

    return pd.DataFrame(records, columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])


def generate_token_overlap_pairs(
    source1: pd.DataFrame,
    candidates: pd.DataFrame,
    token_index: Dict,
    stop_tokens: Set[str],
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

    if not records:
        return pd.DataFrame(columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])

    return pd.DataFrame(records, columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])
