"""
name_cleaner.py — Business name normalisation rules.

Owner: Member A  |  Branch: feature/blocking

TODO (Member A): Implement all functions below.
See TEAM_TASKS.md Task A-2 for the full rules specification.
"""

from __future__ import annotations

import re
from typing import FrozenSet, Tuple

# Legal suffix normalisation map (expand to canonical long form)
LEGAL_SUFFIX_MAP = {
    "corp": "corporation",
    "co": "company",
    "ltd": "limited",
    "pvt": "private",
    "inc": "incorporated",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "plc": "public limited company",
    "gmbh": "gesellschaft mit beschrankter haftung",
    "sarl": "societe a responsabilite limitee",
    "sas": "societe par actions simplifiee",
    "bv": "besloten vennootschap",
    "nv": "naamloze vennootschap",
    "ag": "aktiengesellschaft",
    "sa": "societe anonyme",
}

# Stop-words to remove for token sets (keep them for normalized name, remove for token set)
NAME_STOP_WORDS = {"the", "a", "an", "of", "and", "or", "for", "in", "at", "&"}


def clean_name(name: str) -> Tuple[str, FrozenSet[str], str]:
    """
    Clean and normalise a business name.

    Parameters
    ----------
    name : str
        Raw business name (may be empty).

    Returns
    -------
    normalized : str
        Lightly normalised name (lowercase, whitespace, punctuation cleaned).
    tokens : frozenset[str]
        Set of meaningful tokens (stop-words removed).
    token_sorted : str
        Tokens sorted alphabetically and joined, for order-invariant comparison.
    """
    # TODO: implement
    raise NotImplementedError("clean_name() not yet implemented. See Task A-2.")
