"""
pipeline.py — End-to-end pipeline entry point.

Owner: Member C  |  Branch: feature/evaluation-pipeline

Usage:
    python -m business_entity_resolution.pipeline --mode train --data-dir dataset/train
    python -m business_entity_resolution.pipeline --mode test  --data-dir dataset/test
    python -m business_entity_resolution.pipeline --mode validate --data-dir dataset/train

TODO (Member C): Wire together all modules once A and B deliver their implementations.
See TEAM_TASKS.md Task C-5.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from .config import get_config
from .output_writer import (
    write_matching_results,
    write_candidate_pairs,
    validate_output_integrity,
)
from ..shared.data_loader import (
    load_all_train,
    load_all_test,
    ground_truth_to_dict,
    records_to_dict,
    print_dataset_stats,
)
from ..shared.schemas import COL_ENTITY_ID
from ..evaluation.evaluator import full_evaluation_report, print_report
from ..calibration.calibrator import calibrate_threshold, apply_threshold


def run_pipeline(config: dict, mode: str = "test") -> None:
    """
    Run the full pipeline.

    Parameters
    ----------
    config : dict
        Pipeline configuration. See src/pipeline/config.py.
    mode : str
        "train"    — train model and save artifacts
        "test"     — run inference and write output files
        "validate" — run on train data with ground truth evaluation
    """
    print(f"\n=== Running pipeline in mode: {mode} ===\n")

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Step 1/9: Loading data...")
    if mode in ("train", "validate"):
        s1, s2, s3, gt = load_all_train(config["data_dir_train"])
        print_dataset_stats(s1, s2, s3, gt)
    else:
        s1, s2, s3 = load_all_test(config["data_dir_test"])
        gt = None

    all_s1_ids = s1[COL_ENTITY_ID].tolist()
    s2_ids = set(s2[COL_ENTITY_ID].tolist())
    s3_ids = set(s3[COL_ENTITY_ID].tolist())

    # ------------------------------------------------------------------
    # 2. Normalize
    # ------------------------------------------------------------------
    print("Step 2/9: Normalizing records...")
    try:
        from ..normalization.normalizer import normalize_records
        s1_norm = normalize_records(s1)
        s2_norm = normalize_records(s2)
        s3_norm = normalize_records(s3)
    except NotImplementedError:
        print("  [WARNING] Normalization not yet implemented. Using raw records.")
        s1_norm, s2_norm, s3_norm = s1.copy(), s2.copy(), s3.copy()

    # ------------------------------------------------------------------
    # 3. Generate candidates
    # ------------------------------------------------------------------
    print("Step 3/9: Generating candidate pairs (blocking)...")
    try:
        from ..blocking.blocker import generate_candidates
        from ..candidate_generation.candidate_store import consolidate_candidates
        raw_pairs = generate_candidates(s1_norm, s2_norm, s3_norm, config)
        candidate_pairs = consolidate_candidates(raw_pairs, all_s1_ids, config)
    except NotImplementedError:
        print("  [WARNING] Blocking not yet implemented. Using empty candidates.")
        import pandas as pd
        candidate_pairs = pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "blocking_reasons"]
        )
    print(f"  Generated {len(candidate_pairs)} candidate pairs.")

    # ------------------------------------------------------------------
    # 4. Write candidate_pairs.tsv
    # ------------------------------------------------------------------
    print("Step 4/9: Writing candidate_pairs.tsv...")
    candidate_path = os.path.join(output_dir, config["output"]["candidate_pairs_filename"])
    write_candidate_pairs(candidate_pairs, all_s1_ids, candidate_path)
    print(f"  Written: {candidate_path}")

    # ------------------------------------------------------------------
    # 5. Build features
    # ------------------------------------------------------------------
    print("Step 5/9: Building pairwise features...")
    try:
        from ..features.feature_builder import build_pair_features
        all_records = {}
        for df in [s1_norm, s2_norm, s3_norm]:
            all_records.update(records_to_dict(df))
        feature_df = build_pair_features(candidate_pairs, all_records, config)
    except NotImplementedError:
        print("  [WARNING] Feature builder not yet implemented. Skipping...")
        feature_df = None

    # ------------------------------------------------------------------
    # 6. Apply model
    # ------------------------------------------------------------------
    print("Step 6/9: Running matching model...")
    try:
        from ..models.matcher import predict_proba
        from ..models.gbm_model import GBMMatcherModel
        import pickle
        model_path = os.path.join(config["model_dir"], "gbm_model.pkl")
        if mode == "train":
            from ..models.training_data import build_training_pairs
            gt_dict = ground_truth_to_dict(gt)
            # TODO: Member B implements this
            raise NotImplementedError("Training not yet implemented.")
        else:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No trained model found at {model_path}. Run --mode train first.")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            predictions_df = predict_proba(feature_df, model, config)
    except (NotImplementedError, FileNotFoundError) as e:
        print(f"  [WARNING] Model not available: {e}")
        print("  Using empty predictions (all singletons).")
        import pandas as pd
        predictions_df = pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "match_probability"]
        )

    # ------------------------------------------------------------------
    # 7. Calibrate threshold
    # ------------------------------------------------------------------
    print("Step 7/9: Applying threshold...")
    threshold = config["calibration"]["default_threshold"]
    if gt is not None and predictions_df is not None and len(predictions_df) > 0:
        gt_dict = ground_truth_to_dict(gt)
        result = calibrate_threshold(predictions_df, gt_dict, all_s1_ids)
        threshold = result["best_threshold"]
        print(f"  Best threshold (val): {threshold:.2f}  |  F_0.5: {result['best_f05']:.4f}")
        print(result["threshold_curve"].to_string(index=False))
    else:
        print(f"  Using default threshold: {threshold}")

    # ------------------------------------------------------------------
    # 8. Apply threshold → final predictions
    # ------------------------------------------------------------------
    print("Step 8/9: Generating final predictions...")
    if predictions_df is not None and len(predictions_df) > 0:
        final_predictions = apply_threshold(predictions_df, threshold)
    else:
        final_predictions = {}
    # Ensure all S1 entities have an entry
    for s1_id in all_s1_ids:
        if s1_id not in final_predictions:
            final_predictions[s1_id] = []

    # ------------------------------------------------------------------
    # 9. Write matching_results.tsv
    # ------------------------------------------------------------------
    print("Step 9/9: Writing matching_results.tsv...")
    matching_path = os.path.join(output_dir, config["output"]["matching_results_filename"])
    write_matching_results(final_predictions, all_s1_ids, matching_path)
    print(f"  Written: {matching_path}")

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------
    print("\nRunning internal output validation...")
    violations = validate_output_integrity(
        matching_path, candidate_path, set(all_s1_ids), s2_ids, s3_ids
    )
    if violations:
        print(f"  [FAIL] {len(violations)} violations found:")
        for v in violations:
            print(f"    - {v}")
    else:
        print("  [PASS] All output integrity checks passed.")

    # ------------------------------------------------------------------
    # Evaluation (validate mode only)
    # ------------------------------------------------------------------
    if mode == "validate" and gt is not None:
        print("\nRunning F_0.5 evaluation...")
        gt_dict = ground_truth_to_dict(gt)
        report = full_evaluation_report(final_predictions, gt_dict, candidate_pairs)
        print_report(report)

    print("\n=== Pipeline complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "test", "validate"],
        default="test",
        help="Pipeline mode",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config override file",
    )
    args = parser.parse_args()

    config = get_config()
    if args.data_dir:
        if args.mode in ("train", "validate"):
            config["data_dir_train"] = args.data_dir
        else:
            config["data_dir_test"] = args.data_dir

    if args.config:
        import yaml
        with open(args.config) as f:
            overrides = yaml.safe_load(f)
        from .config import _deep_update
        _deep_update(config, overrides)

    run_pipeline(config, mode=args.mode)


if __name__ == "__main__":
    main()
