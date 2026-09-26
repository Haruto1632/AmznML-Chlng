"""
run_experiments.py — Blocking experiment runner.

Member A | experiments/member_a/blocking/

Runs experiments B0..B7 on a configurable sample of training data.
Records metrics for each strategy combination.

Usage:
    python run_experiments.py                      # default: 10k S1 sample
    python run_experiments.py --sample 50000       # 50k S1 sample
    python run_experiments.py --full               # full training set (slow)
    python run_experiments.py --experiment B1      # single experiment only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# Bootstrap path so we can import shared modules
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "code" / "business_entity_resolution"))

import numpy as np
import pandas as pd

from src.shared.data_loader import load_all_train
from src.shared.schemas import COL_ENTITY_ID
from src.normalization.normalizer import normalize_records
from src.blocking.blocker import generate_candidates
from src.candidate_generation.candidate_store import consolidate_candidates
from src.candidate_generation.blocking_eval import evaluate_blocking, print_blocking_report

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN_DIR = str(_REPO_ROOT / "student_resource" / "dataset" / "train")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Experiment configurations ──────────────────────────────────────────────

BASE_CONFIG = {
    "blocking": {
        "strategy_exact_token": False,
        "strategy_token_overlap": False,
        "strategy_ngram_lsh": False,
        "strategy_phonetic": False,
        "strategy_address_token": False,
        "strategy_embedding_ann": False,
        "ngram_lsh_num_perm": 64,
        "ngram_lsh_threshold": 0.25,
        "ngram_n": 3,
        "token_max_df_frac": 0.01,
    },
    "candidate_generation": {
        "max_candidates_per_s1": None,  # No limit during experiments (measure raw)
        "prefer_high_evidence": True,
    },
}

EXPERIMENTS = {
    "B0": {
        "description": "Baseline: no blocking (empty candidates)",
        "blocking": {},
    },
    "B1": {
        "description": "Token blocking only (exact_country_name_token)",
        "blocking": {"strategy_exact_token": True},
    },
    "B2": {
        "description": "Token + address blocking",
        "blocking": {"strategy_exact_token": True, "strategy_address_token": True},
    },
    "B3": {
        "description": "Token + phonetic blocking",
        "blocking": {"strategy_exact_token": True, "strategy_phonetic": True},
    },
    "B4": {
        "description": "Token + n-gram LSH blocking",
        "blocking": {"strategy_exact_token": True, "strategy_ngram_lsh": True},
    },
    "B5": {
        "description": "Token + address + n-gram LSH",
        "blocking": {
            "strategy_exact_token": True,
            "strategy_address_token": True,
            "strategy_ngram_lsh": True,
        },
    },
    "B6": {
        "description": "Token + address + n-gram LSH + phonetic",
        "blocking": {
            "strategy_exact_token": True,
            "strategy_address_token": True,
            "strategy_ngram_lsh": True,
            "strategy_phonetic": True,
        },
    },
    "B7": {
        "description": "All strategies (excl. ANN — not installed)",
        "blocking": {
            "strategy_exact_token": True,
            "strategy_token_overlap": True,
            "strategy_address_token": True,
            "strategy_ngram_lsh": True,
            "strategy_phonetic": True,
            "strategy_embedding_ann": False,
        },
    },
}


def _make_config(exp_overrides: dict) -> dict:
    """Merge experiment overrides into base config."""
    import copy
    cfg = copy.deepcopy(BASE_CONFIG)
    for k, v in exp_overrides.get("blocking", {}).items():
        cfg["blocking"][k] = v
    return cfg


def run_experiment(
    exp_id: str,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt: pd.DataFrame,
    n_source23: int,
) -> Dict:
    """Run a single blocking experiment and return metrics dict."""
    exp = EXPERIMENTS[exp_id]
    print(f"\n{'='*70}")
    print(f"EXPERIMENT {exp_id}: {exp['description']}")
    print(f"{'='*70}")

    config = _make_config(exp)
    t_start = time.time()

    # Generate candidates
    pairs = generate_candidates(s1, s2, s3, config)

    t_gen = time.time() - t_start

    # Evaluate
    metrics = evaluate_blocking(
        pairs,
        gt,
        n_source1_total=len(s1),
        n_source23_total=n_source23,
    )
    metrics["runtime_seconds"] = t_gen
    metrics["experiment_id"] = exp_id
    metrics["description"] = exp["description"]
    metrics["n_s1"] = len(s1)
    metrics["n_s23"] = n_source23

    print_blocking_report(metrics)
    print(f"  Runtime: {t_gen:.1f}s")

    # Save JSON
    out_path = RESULTS_DIR / f"{exp_id}_results.json"
    with open(out_path, "w") as f:
        # Convert non-serializable types
        safe_metrics = {
            k: (list(v) if isinstance(v, (set, frozenset)) else v)
            for k, v in metrics.items()
            if k != "missed_pairs_sample"
        }
        json.dump(safe_metrics, f, indent=2)
    print(f"  Results saved: {out_path}")

    return metrics


def build_results_table(all_metrics: List[Dict]) -> pd.DataFrame:
    """Build a summary table from all experiment metrics."""
    rows = []
    for m in all_metrics:
        rows.append({
            "experiment": m["experiment_id"],
            "description": m.get("description", ""),
            "blocking_recall": round(m["blocking_recall"], 4),
            "total_candidates": m["total_candidates"],
            "avg_candidates": round(m["avg_candidates_per_s1"], 1),
            "median_candidates": round(m["median_candidates_per_s1"], 1),
            "p95_candidates": round(m["p95_candidates_per_s1"], 1),
            "max_candidates": m["max_candidates_per_s1"],
            "reduction_ratio": round(m["reduction_ratio"], 6)
            if not (isinstance(m["reduction_ratio"], float) and m["reduction_ratio"] != m["reduction_ratio"])
            else "N/A",
            "runtime_s": round(m["runtime_seconds"], 1),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Blocking Experiments")
    parser.add_argument("--sample", type=int, default=10000,
                        help="Number of S1 entities to sample (default: 10000)")
    parser.add_argument("--full", action="store_true",
                        help="Run on full training set (overrides --sample)")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run a single experiment by ID (e.g., B1)")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("BLOCKING EXPERIMENT SUITE — Member A")
    print("="*70)

    # Load data
    print(f"\nLoading training data from: {TRAIN_DIR}")
    t0 = time.time()
    s1_raw, s2_raw, s3_raw, gt = load_all_train(TRAIN_DIR)
    print(f"Loaded in {time.time()-t0:.1f}s: S1={len(s1_raw):,}, S2={len(s2_raw):,}, S3={len(s3_raw):,}")

    # Sampling
    if not args.full and len(s1_raw) > args.sample:
        print(f"\nSampling {args.sample:,} S1 entities (use --full for full run)...")
        # Sample S1 and filter GT to those entities
        s1_sample = s1_raw.sample(n=args.sample, random_state=42).reset_index(drop=True)
        # Keep all S2+S3 (no S2/S3 sampling — index over full candidate pool)
        s1 = s1_sample
        # Filter GT to sampled S1 ids
        sampled_ids = set(s1[COL_ENTITY_ID])
        gt_filtered = gt[gt["source1_entity_id"].isin(sampled_ids)].copy()
    else:
        print(f"\nRunning on full training set ({len(s1_raw):,} S1 entities)...")
        s1 = s1_raw
        gt_filtered = gt

    s2 = s2_raw
    s3 = s3_raw
    n_s23 = len(s2) + len(s3)

    # Normalize (all data — do it once)
    print("\nNormalizing records...")
    t0 = time.time()
    s1_norm = normalize_records(s1)
    s2_norm = normalize_records(s2)
    s3_norm = normalize_records(s3)
    print(f"Normalization done in {time.time()-t0:.1f}s")

    # Determine experiments to run
    if args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"ERROR: Unknown experiment '{args.experiment}'. "
                  f"Valid: {list(EXPERIMENTS.keys())}")
            sys.exit(1)
        exp_ids = [args.experiment]
    else:
        exp_ids = sorted(EXPERIMENTS.keys())

    # Run experiments
    all_metrics = []
    for exp_id in exp_ids:
        try:
            metrics = run_experiment(exp_id, s1_norm, s2_norm, s3_norm, gt_filtered, n_s23)
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Experiment {exp_id} failed: {e}", exc_info=True)

    # Summary table
    if len(all_metrics) > 1:
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        table = build_results_table(all_metrics)
        print(table.to_string(index=False))

        # Save markdown
        md_path = RESULTS_DIR.parent / "blocking_results.md"
        with open(md_path, "w") as f:
            f.write("# Blocking Experiment Results\n\n")
            f.write(f"Sample size: {len(s1):,} S1 entities | "
                    f"Candidate pool: {n_s23:,} S2+S3 records\n\n")
            f.write(table.to_markdown(index=False))
            f.write("\n\n> Target: blocking_recall ≥ 0.98\n")
        print(f"\nMarkdown results saved: {md_path}")


if __name__ == "__main__":
    main()
