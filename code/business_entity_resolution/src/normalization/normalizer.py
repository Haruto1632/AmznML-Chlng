"""
normalizer.py — Entry point for all normalization.

Owner: Member A  |  Branch: feature/blocking

TODO (Member A):
  1. Implement normalize_records()
  2. Wire in name_cleaner.py, address_cleaner.py, country_mapper.py
  3. Add tests in tests/test_normalizer.py
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
          business_name_normalized
          business_name_tokens
          business_name_token_sorted
          business_address_normalized
          address_numbers
          country_normalized
    """
    # TODO: implement using name_cleaner, address_cleaner, country_mapper
    raise NotImplementedError(
        "normalize_records() not yet implemented. "
        "See TEAM_TASKS.md Task A-2 for requirements."
    )
