"""
country_mapper.py — Country normalisation (open-set safe).

Owner: Member A  |  Branch: feature/blocking

IMPORTANT: Country is an OPEN SET. Never hard-code to {US, India}.
France exists in the test set. Unknown countries must pass through unchanged.
"""

from __future__ import annotations

# Known country alias map → canonical lowercase string
# Only add confirmed aliases. Do NOT add speculative ones.
COUNTRY_ALIAS_MAP = {
    # United States
    "usa": "united states",
    "u.s.a.": "united states",
    "u.s.a": "united states",
    "u.s.": "united states",
    "us": "united states",
    "united states of america": "united states",
    "america": "united states",
    # India
    "india": "india",
    "in": "india",
    "bharat": "india",
    # France
    "france": "france",
    "fr": "france",
    "french republic": "france",
    "republique francaise": "france",
    # United Kingdom
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "great britain": "united kingdom",
    "england": "united kingdom",
    # United Arab Emirates
    "uae": "united arab emirates",
    "u.a.e.": "united arab emirates",
    # Germany
    "germany": "germany",
    "deutschland": "germany",
    "de": "germany",
    # Canada
    "canada": "canada",
    "ca": "canada",
    # Australia
    "australia": "australia",
    "au": "australia",
}


def normalize_country(country: str) -> str:
    """
    Normalise a country string.

    Parameters
    ----------
    country : str
        Raw country value (may be empty, abbreviated, or in another language).

    Returns
    -------
    str
        Canonical lowercase country string.
        Returns the original lowercased value if no alias is found.
        NEVER returns None or raises for unknown countries.
    """
    if not country or not country.strip():
        return ""
    cleaned = country.strip().lower()
    return COUNTRY_ALIAS_MAP.get(cleaned, cleaned)
