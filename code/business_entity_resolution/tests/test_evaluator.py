"""
test_evaluator.py — Tests for F_0.5 evaluation logic.

Owner: Member C
Run with: python -m pytest tests/test_evaluator.py -v
"""

import pytest
from ..src.evaluation.evaluator import compute_f05_per_entity, compute_macro_f05, _f05


class TestF05Formula:
    def test_perfect_match(self):
        """Predict exactly the true set → F_0.5 = 1.0"""
        preds = {"S1-00001": ["S2-00001", "S2-00002"]}
        gt = {"S1-00001": ["S2-00001", "S2-00002"]}
        df = compute_f05_per_entity(preds, gt)
        assert df.loc[df["source1_entity_id"] == "S1-00001", "f05"].iloc[0] == pytest.approx(1.0)

    def test_perfect_singleton(self):
        """Predict empty when true is empty → F_0.5 = 1.0"""
        preds = {"S1-00002": []}
        gt = {"S1-00002": []}
        df = compute_f05_per_entity(preds, gt)
        assert df.loc[df["source1_entity_id"] == "S1-00002", "f05"].iloc[0] == pytest.approx(1.0)

    def test_false_positive_singleton(self):
        """Predict non-empty when true is empty → F_0.5 = 0.0"""
        preds = {"S1-00003": ["S3-00001"]}
        gt = {"S1-00003": []}
        df = compute_f05_per_entity(preds, gt)
        assert df.loc[df["source1_entity_id"] == "S1-00003", "f05"].iloc[0] == pytest.approx(0.0)

    def test_worked_example_from_pdf(self):
        """
        PDF example:
          Prediction: [S2-00047, S2-00193, S3-00812]
          Truth:      [S2-00047, S3-00812]
          P = 2/3, R = 2/2 = 1.0
          F_0.5 = (1.25 * 2/3 * 1.0) / (0.25 * 2/3 + 1.0) = 0.7143
        """
        preds = {"S1-00001": ["S2-00047", "S2-00193", "S3-00812"]}
        gt = {"S1-00001": ["S2-00047", "S3-00812"]}
        df = compute_f05_per_entity(preds, gt)
        assert df.loc[df["source1_entity_id"] == "S1-00001", "f05"].iloc[0] == pytest.approx(
            0.7143, abs=1e-3
        )

    def test_zero_prediction(self):
        """Predict empty when truth is non-empty → R=0, P=1 (by convention), F_0.5=0"""
        preds = {"S1-00001": []}
        gt = {"S1-00001": ["S2-00001"]}
        df = compute_f05_per_entity(preds, gt)
        assert df.loc[df["source1_entity_id"] == "S1-00001", "f05"].iloc[0] == pytest.approx(0.0)

    def test_macro_average(self):
        """Macro average is mean of per-entity scores."""
        preds = {"S1-00001": ["S2-00001"], "S1-00002": []}
        gt = {"S1-00001": ["S2-00001"], "S1-00002": []}
        df = compute_f05_per_entity(preds, gt)
        macro = compute_macro_f05(df)
        assert macro == pytest.approx(1.0)

    def test_f05_formula_precision_heavy(self):
        """F_0.5 should be closer to precision than F_1."""
        # P=0.5, R=1.0
        f05 = _f05(0.5, 1.0)
        f1 = (2 * 0.5 * 1.0) / (0.5 + 1.0)  # 0.667
        # F_0.5 should be LESS than F_1 when recall > precision
        assert f05 < f1

    def test_entity_not_in_predictions_treated_as_empty(self):
        """An S1 entity missing from predictions dict should be treated as empty."""
        preds = {}  # no predictions at all
        gt = {"S1-00001": ["S2-00001"]}
        df = compute_f05_per_entity(preds, gt)
        # Empty prediction against non-empty truth → F_0.5 = 0
        assert df.iloc[0]["f05"] == pytest.approx(0.0)
