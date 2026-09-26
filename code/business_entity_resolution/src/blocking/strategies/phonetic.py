"""
phonetic.py — Phonetic blocking using Soundex / Metaphone.

Strategy: pair (S1, S2/S3) if the phonetic code of the first significant
name token matches.

Owner: Member A  |  Branch: feature/blocking

Requires: jellyfish (see requirements.txt)
"""

from __future__ import annotations

import pandas as pd

from ...shared.schemas import STRATEGY_PHONETIC


def _get_phonetic_code(token: str, algorithm: str = "soundex") -> str:
    """Get phonetic code for a single token."""
    try:
        import jellyfish
        if algorithm == "soundex":
            return jellyfish.soundex(token)
        elif algorithm == "metaphone":
            return jellyfish.metaphone(token)
    except ImportError:
        pass
    return token[:1].upper() if token else ""


def run(
    source1: pd.DataFrame,
    candidates: pd.DataFrame,
    algorithm: str = "soundex",
) -> pd.DataFrame:
    """
    Returns DataFrame with columns (source1_entity_id, candidate_entity_id).

    Pairs entities whose first significant name token produces the same
    phonetic code.
    """
    # TODO: implement
    # Approach:
    #   1. For each S2/S3 entity, compute phonetic code of first non-stop name token
    #   2. Build inverted index: phonetic_code → [entity_ids]
    #   3. For each S1 entity, look up its phonetic code in index
    raise NotImplementedError("phonetic.run() not yet implemented.")
