"""
normalizer.py — Entry point for all normalization.

Owner: Member A  |  Branch: feature/blocking

Vectorized implementation:
  - country: fully vectorized via .map()
  - name + address: .apply() row-wise (unavoidable due to tuple returns,
    but single-pass with no repeated work)

Memory: adds ~6 columns to each DataFrame (in-place on copy).
"""

from __future__ import annotations

import pandas as pd

from ..shared.schemas import (
    COL_ENTITY_ID,
    COL_BUSINESS_NAME,
    COL_BUSINESS_ADDRESS,
    COL_COUNTRY,
    COL_NAME_NORMALIZED,
    COL_NAME_TOKENS,
    COL_NAME_TOKEN_SORTED,
    COL_ADDRESS_NORMALIZED,
    COL_ADDRESS_NUMBERS,
    COL_COUNTRY_NORMALIZED,
)
from .name_cleaner import clean_name
from .address_cleaner import clean_address
from .country_mapper import normalize_country


def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all records in a source DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: entity_id, business_name,
        business_address, country.

    Returns
    -------
    pd.DataFrame
        Same rows, with ADDITIONAL columns (originals preserved):
          business_name_normalized    (str)
          business_name_tokens        (frozenset[str])
          business_name_token_sorted  (str)
          business_address_normalized (str)
          address_numbers             (list[str])
          country_normalized          (str)
    """
    result = df.copy()

    # ── Country (fully vectorized) ────────────────────────────────────────
    result[COL_COUNTRY_NORMALIZED] = (
        result[COL_COUNTRY].fillna("").map(normalize_country)
    )

    # ── Business name (.apply — single pass, tuple unpack) ────────────────
    name_results = result[COL_BUSINESS_NAME].fillna("").apply(clean_name)
    result[COL_NAME_NORMALIZED] = name_results.apply(lambda t: t[0])
    result[COL_NAME_TOKENS] = name_results.apply(lambda t: t[1])
    result[COL_NAME_TOKEN_SORTED] = name_results.apply(lambda t: t[2])

    # ── Business address (.apply — single pass, tuple unpack) ─────────────
    addr_results = result[COL_BUSINESS_ADDRESS].fillna("").apply(clean_address)
    result[COL_ADDRESS_NORMALIZED] = addr_results.apply(lambda t: t[0])
    result[COL_ADDRESS_NUMBERS] = addr_results.apply(lambda t: t[1])

    return result
