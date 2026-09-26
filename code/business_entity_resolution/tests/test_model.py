"""
test_model.py — Smoke tests for model training and inference.

Owner: Member B
Run with: python -m pytest tests/test_model.py -v
"""

import pytest
import numpy as np
import pandas as pd

# from ..src.models.gbm_model import GBMMatcherModel
# from ..src.models.matcher import baseline_score
# from ..src.calibration.calibrator import calibrate_threshold, apply_threshold
from ..src.calibration.calibrator import apply_threshold


class TestApplyThreshold:
    """Test threshold application (implemented, testable now)."""

    def test_above_threshold_included(self):
        df = pd.DataFrame([
            {"source1_entity_id": "S1-00001", "candidate_entity_id": "S2-00001", "match_probability": 0.8},
            {"source1_entity_id": "S1-00001", "candidate_entity_id": "S2-00002", "match_probability": 0.4},
        ])
        result = apply_threshold(df, threshold=0.7)
        assert "S2-00001" in result["S1-00001"]
        assert "S2-00002" not in result["S1-00001"]

    def test_empty_prediction_for_all_below_threshold(self):
        df = pd.DataFrame([
            {"source1_entity_id": "S1-00001", "candidate_entity_id": "S2-00001", "match_probability": 0.3},
        ])
        result = apply_threshold(df, threshold=0.7)
        assert result.get("S1-00001", []) == []

    def test_entity_not_in_df_gets_no_entry(self):
        df = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "match_probability"])
        result = apply_threshold(df, threshold=0.7)
        assert result == {}


class TestBaselineScore:
    # def test_high_similarity_high_score(self):
    #     row = {
    #         "name_token_jaccard": 0.9,
    #         "addr_token_jaccard": 0.8,
    #         "country_exact_normalized": 1.0,
    #     }
    #     score = baseline_score(row)
    #     assert score > 0.8
    #
    # def test_zero_similarity_zero_score(self):
    #     row = {
    #         "name_token_jaccard": 0.0,
    #         "addr_token_jaccard": 0.0,
    #         "country_exact_normalized": 0.0,
    #     }
    #     score = baseline_score(row)
    #     assert score == pytest.approx(0.0)
    pass
