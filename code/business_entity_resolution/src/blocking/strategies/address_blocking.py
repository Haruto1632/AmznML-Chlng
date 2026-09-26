"""
address_blocking.py — Address-component blocking strategies.

Owner: Member A  |  Branch: feature/blocking

Strategy: address_token_overlap
  Matches S1 with candidates sharing:
    (a) address_number_token: same numeric token from address
    (b) address_number + name_token: shared address number AND shared name token
        (more precise, fewer false positives)

Design notes:
  - Extract numeric tokens from addresses (building numbers, postal codes)
  - Build inverted index: number → set(candidate_ids)
  - Build joint index: (number, name_token) → set(candidate_ids) for precision
  - Postal codes are high-cardinality discriminators in India/US; street numbers
    are useful in France
  - No external geocoding used
  - Purely derived from supplied address fields

IDF filtering:
  - Very common numbers (e.g. "1", "100") filtered by max_df_frac
  - Single-digit numbers excluded
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Dict, Set, Tuple

import pandas as pd

from ...shared.schemas import (
    COL_ENTITY_ID,
    COL_NAME_TOKENS,
    COL_COUNTRY_NORMALIZED,
    COL_ADDRESS_NUMBERS,
    COL_ADDRESS_NORMALIZED,
    COL_SOURCE1_ID,
    COL_CANDIDATE_ID,
    STRATEGY_ADDRESS_TOKEN,
)
logger = logging.getLogger(__name__)


def build_address_index(
    candidates: pd.DataFrame,
    max_df_frac: float = 0.005,
    min_number_len: int = 3,
) -> Tuple[Dict, Dict]:
    """
    Build two address inverted indices.

    Parameters
    ----------
    candidates : pd.DataFrame
        S2+S3 normalized with columns:
        entity_id, address_numbers, business_name_tokens, address_normalized
    max_df_frac : float
        Numbers in > this fraction of candidates are stop-numbers.
    min_number_len : int
        Minimum length of number token to include (filter out 1-2 digit noise).

    Returns
    -------
    number_index : dict
        address_number → set(entity_ids)
    number_name_index : dict
        (address_number, name_token) → set(entity_ids)
        More precise; requires both number AND name token to match.
    """
    n_cands = len(candidates)
    max_df = max(1, int(n_cands * max_df_frac))

    # Compute number document frequency
    from collections import Counter
    num_counter: Counter = Counter()
    for row in candidates.itertuples(index=False):
        for num in (row.address_numbers or []):
            if len(num) >= min_number_len:
                num_counter[num] += 1

    stop_numbers = {num for num, cnt in num_counter.items() if cnt > max_df}
    logger.info(f"Address stop-numbers: {len(stop_numbers)} excluded (df > {max_df})")

    number_index: Dict[str, Set[str]] = defaultdict(set)
    number_name_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for row in candidates.itertuples(index=False):
        eid = row.entity_id
        numbers = row.address_numbers or []
        name_tokens = row.business_name_tokens  # frozenset

        for num in numbers:
            if len(num) < min_number_len or num in stop_numbers:
                continue
            number_index[num].add(eid)

            # Cross-index with name tokens for precision
            for tok in name_tokens:
                if len(tok) >= 3:  # only substantive name tokens
                    number_name_index[(num, tok)].add(eid)

    logger.info(
        f"Address index: {len(number_index):,} number keys, "
        f"{len(number_name_index):,} (number, name_token) keys"
    )
    return dict(number_index), dict(number_name_index)


def generate_address_pairs(
    source1: pd.DataFrame,
    number_index: Dict,
    number_name_index: Dict,
    min_number_len: int = 3,
    use_joint_index: bool = True,
) -> pd.DataFrame:
    """
    Strategy: address_token_overlap
    Generates pairs sharing:
      - An address number (if use_joint_index=False)
      - An address number AND a name token (if use_joint_index=True, more precise)

    Returns DataFrame: source1_entity_id, candidate_entity_id
    """
    records = []

    for row in source1.itertuples(index=False):
        s1_id = row.entity_id
        numbers = row.address_numbers or []
        name_tokens = row.business_name_tokens  # frozenset

        seen: Set[str] = set()

        if use_joint_index:
            # More precise: require address number + name token match
            for num in numbers:
                if len(num) < min_number_len:
                    continue
                for tok in name_tokens:
                    if len(tok) >= 3:
                        key = (num, tok)
                        for cand_id in number_name_index.get(key, ()):
                            if cand_id not in seen:
                                seen.add(cand_id)
                                records.append((s1_id, cand_id))
        else:
            # Less precise: address number only
            for num in numbers:
                if len(num) < min_number_len:
                    continue
                for cand_id in number_index.get(num, ()):
                    if cand_id not in seen:
                        seen.add(cand_id)
                        records.append((s1_id, cand_id))

    if not records:
        return pd.DataFrame(columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])

    return pd.DataFrame(records, columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])
