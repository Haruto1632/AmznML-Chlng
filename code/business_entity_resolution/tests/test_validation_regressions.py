"""Regression tests for complete-cohort evaluation and calibration."""
import pandas as pd
import pytest

from ..src.calibration.calibrator import apply_threshold, calibrate_threshold
from ..src.candidate_generation.blocking_eval import evaluate_blocking
from ..src.evaluation.evaluator import full_evaluation_report
from ..src.shared.data_loader import load_source, ground_truth_to_dict

PAIR_COLS = ["source1_entity_id", "candidate_entity_id"]


def test_blocking_counts_empty_entities_and_unique_pairs():
    pairs = pd.DataFrame([("a", "S2-1"), ("a", "S2-1")], columns=PAIR_COLS)
    gt = pd.DataFrame({"source1_entity_id": ["a", "b", "c"],
                       "matched_entity_ids": ["S2-1", "S3-1", ""]})
    report = evaluate_blocking(pairs, gt, 3, 10)
    assert report["blocking_recall"] == .5
    assert report["total_candidates"] == 1
    assert report["avg_candidates_per_s1"] == pytest.approx(1 / 3)
    assert report["median_candidates_per_s1"] == 0
    assert report["reduction_ratio"] == pytest.approx(29 / 30)


def test_report_includes_missed_entity_and_singleton():
    pairs = pd.DataFrame([("a", "S2-1")], columns=PAIR_COLS)
    report = full_evaluation_report({"a": ["S2-1"]},
                                   {"a": ["S2-1"], "b": ["S3-1"], "c": []}, pairs)
    assert report["macro_f05"] == pytest.approx(2 / 3)
    assert report["singleton_accuracy"] == 1
    assert report["blocking_recall"] == .5


def test_threshold_deduplicates_and_ties_choose_conservative_threshold():
    scores = pd.DataFrame([("a", "S2-1", .9)] * 2,
                          columns=PAIR_COLS + ["match_probability"])
    assert apply_threshold(scores, .5) == {"a": ["S2-1"]}
    result = calibrate_threshold(scores, {"a": ["S2-1"], "b": []}, ["a", "b"], [.5, .8])
    assert result["best_threshold"] == .8
    assert result["best_f05"] == 1


def test_calibration_keeps_zero_candidate_false_negatives():
    scores = pd.DataFrame(columns=PAIR_COLS + ["match_probability"])
    result = calibrate_threshold(scores, {"a": ["S2-1"], "b": []}, ["a", "b"])
    assert result["best_f05"] == .5
    with pytest.raises(ValueError, match="exactly"):
        calibrate_threshold(scores, {"a": []}, ["a", "b"])
    with pytest.raises(ValueError, match="non-empty"):
        full_evaluation_report({}, {})


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -1, 1.1])
def test_invalid_probabilities_rejected(probability):
    scores = pd.DataFrame([("a", "S2-1", probability)], columns=PAIR_COLS + ["match_probability"])
    with pytest.raises(ValueError):
        apply_threshold(scores, .5)


def test_literal_na_is_not_a_missing_business_name(tmp_path):
    path = tmp_path / "source.tsv"
    path.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tNA\t\tNULL\n", encoding="utf-8")
    row = load_source(path).iloc[0]
    assert row.business_name == "NA"
    assert row.country == "NULL"
    assert row.business_address == ""


def test_normalization_preserves_originals_and_unknown_country():
    from ..src.normalization.normalizer import normalize_records
    frame = pd.DataFrame({"entity_id": ["S1-1"], "business_name": ["Acme & Co."],
                          "business_address": ["12 Rue Victor"], "country": ["Atlantis"]})
    original = frame.copy(deep=True)
    normalized = normalize_records(frame)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(normalized[frame.columns], original)
    assert normalized.country_normalized.iloc[0] == "atlantis"
    assert "12" in normalized.address_numbers.iloc[0]


def test_shared_blocking_and_consolidation_contract():
    from ..src.normalization.normalizer import normalize_records
    from ..src.blocking.blocker import generate_candidates
    from ..src.candidate_generation.candidate_store import consolidate_candidates
    from ..src.pipeline.config import get_config
    columns = ["entity_id", "business_name", "business_address", "country"]
    s1 = normalize_records(pd.DataFrame([("S1-1", "Acme", "12 Rue Victor", "France"),
                                         ("S1-2", "Zebra", "", "France")], columns=columns))
    s2 = normalize_records(pd.DataFrame([("S2-1", "Acme", "12 Rue Victor", "France")], columns=columns))
    s3 = normalize_records(pd.DataFrame([("S3-1", "Acme", "13 Rue Victor", "France")], columns=columns))
    config = get_config()
    for key in config["blocking"]:
        if key.startswith("strategy_"):
            config["blocking"][key] = key == "strategy_exact_token"
    config["blocking"]["token_max_df_frac"] = 1.0
    config["candidate_generation"]["max_candidates_per_s1"] = 1
    pairs = generate_candidates(s1, s2, s3, config)
    assert set(pairs.candidate_entity_id) == {"S2-1", "S3-1"}
    final = consolidate_candidates(pairs, s1.entity_id.tolist(), config)
    assert len(final) == 1
    assert final.source1_entity_id.tolist() == ["S1-1"]
    assert final.blocking_reasons.iloc[0]


def test_duplicate_ground_truth_rejected():
    gt = pd.DataFrame({"source1_entity_id": ["a", "a"], "matched_entity_ids": ["", ""]})
    with pytest.raises(ValueError, match="Duplicate"):
        ground_truth_to_dict(gt)


def test_candidate_integrity_rejects_extra_and_duplicate_rows(tmp_path):
    from ..src.pipeline.output_writer import validate_output_integrity
    matching = tmp_path / "matching.tsv"
    candidates = tmp_path / "candidates.tsv"
    matching.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\t\n", encoding="utf-8")
    candidates.write_text("source1_entity_id\tcandidate_entity_ids\nS1-1\t\nS1-1\t\nS1-2\t\n", encoding="utf-8")
    errors = validate_output_integrity(str(matching), str(candidates), {"S1-1"}, set(), set())
    assert any("extra" in error for error in errors)
    assert any("duplicate" in error for error in errors)


def test_baseline_split_includes_no_candidate_entities_and_does_not_sample_validation():
    import sys
    from pathlib import Path
    experiment = Path(__file__).resolve().parents[3] / "experiments/member_a/baseline_v1"
    sys.path.insert(0, str(experiment))
    from pipeline import BaselinePipeline
    from run_experiment import load_config
    from sklearn.model_selection import train_test_split

    config = load_config()
    config["training"]["sample_train_pairs"] = 10
    pipeline = BaselinePipeline(config)
    ids = [f"S1-{i}" for i in range(30)]
    _, val_ids = train_test_split(ids, test_size=.2, random_state=42)
    no_candidates = val_ids[0]
    pipeline.s1_train = pd.DataFrame({"entity_id": ids})
    pipeline.gt_train = pd.DataFrame({"source1_entity_id": ids,
                                      "matched_entity_ids": ["S2-1"] * len(ids)})
    features = pd.DataFrame([(sid, cid, value) for sid in ids if sid != no_candidates
                              for cid, value in [("S2-1", 1.), ("S2-2", 0.)]],
                             columns=PAIR_COLS + ["similarity"])
    train, val, _, _, actual_val_ids = pipeline._prepare_training_data(features)
    assert actual_val_ids == set(val_ids)
    assert no_candidates in actual_val_ids
    assert len(val) == 10
    assert len(train) == 10
    assert not set(train.index.get_level_values(0)) & actual_val_ids
