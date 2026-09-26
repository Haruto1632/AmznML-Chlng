"""
run_experiment.py — Entry point for baseline experiment v1.

Member A — Baseline v1

Usage
-----
Run from the repository root:

    # Use auto-detected default dataset root (student_resource/dataset/)
    python experiments/member_a/baseline_v1/run_experiment.py

    # Override the dataset root explicitly (recommended)
    python experiments/member_a/baseline_v1/run_experiment.py \\
        --data-dir "student_resource/dataset"

    # Separate overrides for train and test
    python experiments/member_a/baseline_v1/run_experiment.py \\
        --train-dir "student_resource/dataset/train" \\
        --test-dir  "student_resource/dataset/test"

    # Custom config file
    python experiments/member_a/baseline_v1/run_experiment.py \\
        --config experiments/member_a/baseline_v1/experiment_config.yaml \\
        --data-dir "student_resource/dataset"

Path resolution
---------------
All paths are resolved relative to the repository root (the parent of the
experiments/ directory) so the command works regardless of which directory
you are in when you run it.
"""

from __future__ import annotations

# ── Path bootstrap — must be the very first project import ─────────────────
import _project_paths  # noqa: F401

# ── Stdlib ─────────────────────────────────────────────────────────────────
import argparse
import sys
from pathlib import Path

# ── Third-party ────────────────────────────────────────────────────────────
import pandas as pd
import yaml

# ── Pipeline ───────────────────────────────────────────────────────────────
from pipeline import BaselinePipeline


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path | None = None) -> dict:
    """
    Load YAML config from *config_path*.
    Defaults to experiment_config.yaml next to this file.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "experiment_config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        sys.exit(f"ERROR: config file not found: {config_path}")
    with config_path.open() as fh:
        return yaml.safe_load(fh)


def apply_data_dir_overrides(config: dict, args: argparse.Namespace) -> dict:
    """
    Merge CLI data-directory arguments into *config*, overriding the YAML
    values.  The precedence is:

        explicit --train-dir / --test-dir  >  --data-dir  >  YAML config

    All paths are resolved to absolute paths based on the repository root so
    the pipeline does not care about the caller's working directory.
    """
    repo_root = _project_paths.REPO_ROOT

    def _resolve(p: str | Path) -> str:
        """Return an absolute string path, resolving relative paths from repo root."""
        p = Path(p)
        if p.is_absolute():
            return str(p)
        return str(repo_root / p)

    # Start from config values (may still be relative)
    train_dir = config["data"]["train_dir"]
    test_dir  = config["data"].get("test_dir", "")
    out_dir   = config["data"]["output_dir"]

    # --data-dir overrides both train and test
    if args.data_dir:
        base = Path(args.data_dir)
        train_dir = base / "train"
        test_dir  = base / "test"

    # --train-dir / --test-dir override individually (highest priority)
    if args.train_dir:
        train_dir = args.train_dir
    if args.test_dir:
        test_dir = args.test_dir

    config["data"]["train_dir"] = _resolve(train_dir)
    config["data"]["test_dir"]  = _resolve(test_dir)
    config["data"]["output_dir"] = _resolve(out_dir)

    return config


# ---------------------------------------------------------------------------
# Results report
# ---------------------------------------------------------------------------

def write_results_report(pipeline: BaselinePipeline, output_path: Path) -> None:
    """Write a human-readable markdown experiment report."""
    results = pipeline.results
    config  = pipeline.config

    singleton_rate = (
        results.get("singleton_count", 0) / results.get("total_entities", 1)
        if results.get("total_entities", 0) > 0 else 0.0
    )

    report = f"""\
# Baseline Experiment v1 — Results

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Experiment:** {config['experiment']['name']}  
**Description:** {config['experiment']['description']}

---

## Configuration

### Data paths
```
train_dir  : {config['data']['train_dir']}
test_dir   : {config['data'].get('test_dir', '(not set)')}
output_dir : {config['data']['output_dir']}
```

### Normalization
```yaml
{yaml.dump(config['normalization'], default_flow_style=False).rstrip()}
```

### Blocking
```yaml
{yaml.dump(config['blocking'], default_flow_style=False).rstrip()}
```

### Features
```yaml
{yaml.dump(config['features'], default_flow_style=False).rstrip()}
```

### Model
```yaml
{yaml.dump(config['model'], default_flow_style=False).rstrip()}
```

### Training
```yaml
{yaml.dump(config['training'], default_flow_style=False).rstrip()}
```

---

## Results

### Core metrics

| Metric | Value |
|---|---|
| **Macro F_0.5** | **{results.get('macro_f05', 0):.4f}** |
| Macro Precision | {results.get('macro_precision', 0):.4f} |
| Macro Recall | {results.get('macro_recall', 0):.4f} |
| Singleton Accuracy | {results.get('singleton_accuracy', 0) or 0:.4f} |
| Non-Singleton F_0.5 | {results.get('non_singleton_f05', 0) or 0:.4f} |
| Best Threshold | {results.get('best_threshold', 0):.2f} |

### Candidate / blocking statistics

| Metric | Value |
|---|---|
| Blocking Recall | {results.get('blocking_recall', 0):.4f} |
| Total Candidates | {results.get('total_candidates', 0):,} |
| Avg Candidates / S1 | {results.get('avg_candidates', 0):.1f} |
| Reduction Ratio | {results.get('reduction_ratio', 0):.6f} |

### Dataset statistics

| Metric | Value |
|---|---|
| S1 Entities (val set) | {results.get('total_entities', 0):,} |
| Singletons | {results.get('singleton_count', 0):,} |
| Singleton Rate | {singleton_rate:.2%} |

### Runtime

| | |
|---|---|
| Total runtime | {results.get('runtime_seconds', 0):.1f} s ({results.get('runtime_seconds', 0)/60:.1f} min) |

---

## Limitations / next steps

1. Analyse false negatives: which true matches did blocking miss?
2. Analyse false positives: what makes confusing non-matches?
3. Phonetic blocking is Latin-only — Devanagari names rely on transliteration quality.
4. TF-IDF vocabulary is corpus-dependent; test-set generalisation untested.
5. No cross-encoder or graph refinement yet.
6. Threshold is global; per-evidence-level thresholds may improve precision.

---

*Experiment completed successfully.*
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nResults report → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run baseline entity-resolution experiment v1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to YAML config file. "
             "Default: experiments/member_a/baseline_v1/experiment_config.yaml",
    )
    p.add_argument(
        "--data-dir",
        metavar="DIR",
        default=None,
        help="Dataset root containing train/ and test/ sub-directories. "
             "Overrides config train_dir / test_dir. "
             "Example: student_resource/dataset",
    )
    p.add_argument(
        "--train-dir",
        metavar="DIR",
        default=None,
        help="Path to the train/ directory (overrides --data-dir for training). "
             "Example: student_resource/dataset/train",
    )
    p.add_argument(
        "--test-dir",
        metavar="DIR",
        default=None,
        help="Path to the test/ directory (overrides --data-dir for test). "
             "Example: student_resource/dataset/test",
    )
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Load base config then apply any CLI overrides
    config = load_config(args.config)
    config = apply_data_dir_overrides(config, args)

    print("Resolved data paths:")
    print(f"  train_dir  = {config['data']['train_dir']}")
    print(f"  test_dir   = {config['data'].get('test_dir', '(not set)')}")
    print(f"  output_dir = {config['data']['output_dir']}")

    # Validate that train_dir actually exists before spending time on anything
    from pathlib import Path as _Path
    train_dir = _Path(config["data"]["train_dir"])
    if not train_dir.exists():
        sys.exit(
            f"\nERROR: train_dir does not exist: {train_dir}\n"
            f"Pass the correct path with --data-dir or --train-dir.\n"
            f"Expected structure:\n"
            f"  <dir>/train_source1.tsv\n"
            f"  <dir>/train_source2.tsv\n"
            f"  <dir>/train_source3.tsv\n"
            f"  <dir>/train_ground_truth.tsv"
        )

    # Run
    pipeline = BaselinePipeline(config)
    results  = pipeline.run()

    # Write markdown report
    report_path = Path(__file__).parent / "RESULTS.md"
    write_results_report(pipeline, report_path)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"  Macro F_0.5      : {results['macro_f05']:.4f}")
    print(f"  Blocking recall  : {results['blocking_recall']:.4f}")
    print(f"  Avg cands / S1   : {results['avg_candidates']:.1f}")
    print(f"  Runtime          : {results['runtime_seconds']:.0f}s"
          f" ({results['runtime_seconds']/60:.1f} min)")
    print(f"  Results report   : {report_path}")


if __name__ == "__main__":
    main()
