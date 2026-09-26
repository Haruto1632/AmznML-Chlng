"""
test_features.py — Tests for pairwise feature computation.

Owner: Member B
Run with: python -m pytest tests/test_features.py -v

NOTE: These are stub tests. Fill in when features are implemented.
"""

import pytest
import pandas as pd

# from ..src.features.name_features import compute_name_features
# from ..src.features.address_features import compute_address_features
# from ..src.features.country_features import compute_country_features


class TestNameFeatures:
    """Tests for name similarity features."""

    # def test_exact_normalized_equality(self):
    #     feats = compute_name_features("amazon inc", "amazon inc", ...)
    #     assert feats["name_exact_normalized"] == 1.0
    #
    # def test_different_names_low_similarity(self):
    #     feats = compute_name_features("amazon inc", "walmart corp", ...)
    #     assert feats["name_token_jaccard"] < 0.3
    #
    # def test_abbreviation_match(self):
    #     feats = compute_name_features("IBM", "International Business Machines", ...)
    #     assert feats["name_abbrev_score"] > 0.5
    #
    # def test_no_nan_features(self):
    #     """All features must be numeric (no NaN, no None)."""
    #     feats = compute_name_features("", "test", ...)
    #     for k, v in feats.items():
    #         assert v is not None and not (v != v)  # not NaN
    pass


class TestAddressFeatures:
    # def test_exact_address_match(self):
    #     pass
    #
    # def test_both_empty_flag(self):
    #     pass
    #
    # def test_numeric_overlap(self):
    #     pass
    pass


class TestCountryFeatures:
    # def test_exact_country_match(self):
    #     pass
    #
    # def test_country_mismatch(self):
    #     pass
    pass
