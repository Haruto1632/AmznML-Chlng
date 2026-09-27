"""Bounded real-data smoke run, NOT a representative quality benchmark.

Read a small S1 prefix, stream the TSVs to retain its true matches and a bounded
distractor prefix. Run the existing baseline and validate its validation-only
outputs. No rows are invented. Memory is bounded by one input chunk plus sample.
"""
import _project_paths
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml

import pandas as pd

from run_experiment import load_config
from pipeline import BaselinePipeline
from src.shared.data_loader import ground_truth_to_dict
from src.pipeline.output_writer import validate_output_integrity


def select_rows(path, ids, column, distractors=0):
    kept = []
    seen = 0
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=100_000):
        mask = chunk[column].isin(ids)
        prefix = max(0, min(len(chunk), distractors - seen))
        mask.iloc[:prefix] = True
        kept.append(chunk.loc[mask])
        seen += len(chunk)
    return pd.concat(kept, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", type=int, default=100)
    parser.add_argument("--distractors", type=int, default=200)
    parser.add_argument("--data-dir", type=Path, default=_project_paths.DEFAULT_TRAIN_DIR)
    parser.add_argument("--output-dir", type=Path, default=_project_paths.REPO_ROOT / "output" / "smoke")
    args = parser.parse_args()
    if not 20 <= args.entities <= 1000 or not 0 <= args.distractors <= 5000:
        parser.error("Smoke bounds: 20..1000 entities; 0..5000 distractors per source")
    started = time.perf_counter()
    train = args.output_dir / "sample_train"
    validation = args.output_dir / "validation_sources"
    train.mkdir(parents=True, exist_ok=True)
    validation.mkdir(parents=True, exist_ok=True)
    s1 = pd.read_csv(args.data_dir / "train_source1.tsv", sep="\t", dtype=str,
                     keep_default_na=False, nrows=args.entities)
    gt = select_rows(args.data_dir / "train_ground_truth.tsv", set(s1.entity_id), "source1_entity_id")
    truth = ground_truth_to_dict(gt)
    needed = {mid for mids in truth.values() for mid in mids}
    print(f"Selecting {len(s1)} real S1 rows, {len(needed)} true matches and distractors", flush=True)
    sources = {1: s1}
    for source in (2, 3):
        sources[source] = select_rows(args.data_dir / f"train_source{source}.tsv", needed, "entity_id", args.distractors)
    available = set(sources[2].entity_id) | set(sources[3].entity_id)
    if needed - available:
        raise ValueError("Sample ground truth references missing source records")
    for source, frame in sources.items():
        frame.to_csv(train / f"train_source{source}.tsv", sep="\t", index=False)
    gt.to_csv(train / "train_ground_truth.tsv", sep="\t", index=False)
    config = load_config()
    config["data"]["train_dir"] = str(train)
    config["data"]["output_dir"] = str(args.output_dir)
    config["model"].update(n_estimators=30, early_stopping_rounds=5, min_child_samples=5, num_threads=2)
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    pipeline = BaselinePipeline(config)
    results = pipeline.run()
    matching = args.output_dir / "matching_results.tsv"
    candidates = args.output_dir / "candidate_pairs.tsv"
    val_ids = set(pd.read_csv(matching, sep="\t", dtype=str).source1_entity_id)
    sources[1][sources[1].entity_id.isin(val_ids)].to_csv(validation / "test_source1.tsv", sep="\t", index=False)
    for source in (2, 3):
        sources[source].to_csv(validation / f"test_source{source}.tsv", sep="\t", index=False)
    violations = validate_output_integrity(str(matching), str(candidates), val_ids,
                                           set(sources[2].entity_id), set(sources[3].entity_id))
    if violations:
        raise ValueError(violations)
    subprocess.run([sys.executable, str(_project_paths.REPO_ROOT / "student_resource/utils/validate_submission.py"),
                    "--matching", str(matching), "--candidate", str(candidates),
                    "--test-dir", str(validation), "--check-ids"], check=True)
    results["scope"] = "real-data enriched smoke subset; calibration score, not generalization estimate"
    results["sample_selection_seconds"] = time.perf_counter() - started - results["runtime_seconds"]
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
