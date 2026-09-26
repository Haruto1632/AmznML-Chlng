"""
test_normalizer.py — Tests for normalization rules.

Owner: Member A
Run with: python -m pytest tests/test_normalizer.py -v

NOTE: Tests are written before implementation (TDD approach).
Add tests as you implement features. Do NOT remove tests.
"""

import pytest
from ..src.normalization.country_mapper import normalize_country

# NOTE: Once normalizer.py is implemented, add:
# from ..src.normalization.normalizer import normalize_records
# from ..src.normalization.name_cleaner import clean_name
# from ..src.normalization.address_cleaner import clean_address


class TestCountryMapper:
    """country_mapper.py is already implemented — test it now."""

    def test_us_aliases(self):
        assert normalize_country("USA") == "united states"
        assert normalize_country("U.S.A.") == "united states"
        assert normalize_country("united states of america") == "united states"

    def test_france_aliases(self):
        assert normalize_country("France") == "france"
        assert normalize_country("FR") == "france"
        assert normalize_country("FRANCE") == "france"

    def test_india(self):
        assert normalize_country("India") == "india"
        assert normalize_country("INDIA") == "india"

    def test_unknown_country_passes_through(self):
        """Unknown countries must NOT crash or return None."""
        result = normalize_country("Ruritania")
        assert result == "ruritania"

    def test_empty_country(self):
        assert normalize_country("") == ""
        assert normalize_country("   ") == ""

    def test_none_equivalent(self):
        assert normalize_country("") == ""


class TestNameCleaner:
    """
    Tests for name_cleaner.clean_name().
    Add more tests as you implement the function.
    """
    # Uncomment when implemented:
    # def test_lowercase(self):
    #     norm, tokens, sorted_ = clean_name("AMAZON INC")
    #     assert norm == "amazon inc"
    #
    # def test_ampersand_replacement(self):
    #     norm, _, _ = clean_name("Smith & Jones")
    #     assert "and" in norm
    #     assert "&" not in norm
    #
    # def test_legal_suffix_normalisation(self):
    #     norm1, tokens1, _ = clean_name("Acme Corp")
    #     norm2, tokens2, _ = clean_name("Acme Corporation")
    #     # After suffix normalisation, tokens should overlap
    #     assert "corporation" in tokens1 or "corp" in tokens1
    #
    # def test_original_not_modified(self):
    #     # This tests that the DataFrame approach preserves originals
    #     import pandas as pd
    #     from ..src.normalization.normalizer import normalize_records
    #     df = pd.DataFrame([{
    #         "entity_id": "S1-00001",
    #         "business_name": "Acme Corp",
    #         "business_address": "123 Main St",
    #         "country": "USA"
    #     }])
    #     result = normalize_records(df)
    #     assert result["business_name"].iloc[0] == "Acme Corp"  # original preserved
    pass


class TestAddressCleaner:
    """Tests for address_cleaner.clean_address(). Add as you implement."""
    # def test_street_abbreviation(self):
    #     from ..src.normalization.address_cleaner import clean_address
    #     norm, numbers = clean_address("123 Main St")
    #     assert "street" in norm
    #
    # def test_number_extraction(self):
    #     norm, numbers = clean_address("42 Rue de Rivoli")
    #     assert "42" in numbers
    #
    # def test_french_terms_preserved(self):
    #     norm, _ = clean_address("15 Rue de la Paix, Paris")
    #     assert "rue" in norm  # French term preserved, not translated
    pass
