"""
address_cleaner.py — Business address normalisation rules.

Owner: Member A  |  Branch: feature/blocking

TODO (Member A): Implement all functions below.
See TEAM_TASKS.md Task A-2 for the full rules specification.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# English address abbreviation expansion map
ADDRESS_ABBREV_MAP = {
    "st": "street",
    "rd": "road",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "hwy": "highway",
    "fwy": "freeway",
    "pkwy": "parkway",
    "expy": "expressway",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    # Indian address terms (keep as-is but expand common ones)
    "nagar": "nagar",
    "marg": "marg",
    # French address terms — KEEP, do not translate to English
    # rue, avenue, boulevard, cedex, arrondissement → preserved
}


def clean_address(address: str) -> Tuple[str, List[str]]:
    """
    Clean and normalise a business address.

    Parameters
    ----------
    address : str
        Raw business address (may be empty).

    Returns
    -------
    normalized : str
        Normalised address string.
    numbers : list[str]
        Extracted numeric tokens (building numbers, PIN codes, etc.).
    """
    # TODO: implement
    raise NotImplementedError("clean_address() not yet implemented. See Task A-2.")
