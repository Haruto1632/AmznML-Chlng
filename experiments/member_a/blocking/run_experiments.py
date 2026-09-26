"""
run_experiments.py — Blocking experiment runner.

Member A | experiments/member_a/blocking/

Runs experiments B0..B7 on a configurable sample of training data.
Uses pre-normalized parquet cache when available (run preprocess_data.py first).

Usage:
    python run_experiments.py                      # default: all experiments, full S1
    python run_experiments.py --sample 10000       # 10k S1 sample
    python run_experiments.py --experiment B1      # single experiment only
    python run_experiments.py --quick              # B0,B1,B2 only (fast sanity check)
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

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "code" / "business_entity_resolution"))

import numpy as np
import pandas as pd

from src.shared.data_loader import load_all_train, load_source
from src.shared.schemas import COL_ENTITY_ID
from src.normalization.normalizer import normalize_records
from src.blocking.blocker import generate_candidates
from src.candidate_generation.candidate_store import consolidate_candidates
from src.candidate_generation.blocking_eval import evaluate_blocking, print_blocking_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

TRAIN_DIR = str(_REPO_ROOT / "student_resource" / "dataset" / "train")
RESULTS_DIR = Path(__file__).parent / "results"
CACHE_DIR = Path(__file__).parent / "cache"
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
        "token_max_df_frac": 0.001,
        "token_max_bucket_size": 5000,
    },
    "candidate_generation": {
        "max_candidates_per_s1": None,
        "prefer_high_evidence": True,
    },
}

EXPERIMENTS = {
    "B0": {
        "description": "Baseline: no blocking (empty candidates) — recall = 0",
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
        "description": "All strategies (excl. ANN — faiss not installed)",
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
    import copy
    cfg = copy.deepcopy(BASE_CONFIG)
    for k, v in exp_overrides.get("blocking", {}).items():
        cfg["blocking"][k] = v
    return cfg


def _load_from_cache(cache_path: Path) -> pd.DataFrame:
    """Load pre-normalized parquet and reconstruct Python types."""
    df = pd.read_parquet(cache_path)
    df["business_name_tokens"] = df["business_name_tokens_str"].apply(
        lambda s: frozenset(s.split("|")) if s else frozenset()
    )
    df["address_numbers"] = df["address_numbers_str"].apply(
        lambda s: s.split("|") if s else []
    )
    return df


def load_normalized_data(sample_s1: int = None):
    """
    Load normalized data. Uses parquet cache if available, otherwise normalizes.

    Parameters
    ----------
    sample_s1 : int or None
        If set, take only this many S1 entities (random sample).
        Full S2+S3 is always used.
    """
    print("\nLoading normalized data...")
    t0 = time.time()

    # Check for cache
    s1_cache = CACHE_DIR / "s1_train_norm.parquet"
    s2_cache = CACHE_DIR / "s2_train_norm.parquet"
    s3_cache = CACHE_DIR / "s3_train_norm.parquet"

    cache_available = s1_cache.exists() and s2_cache.exists() and s3_cache.exists()

    if cache_available:
        print("  Using pre-normalized parquet cache...")
        s1_full = _load_from_cache(s1_cache)
        s2 = _load_from_cache(s2_cache)
        s3 = _load_from_cache(s3_cache)
        print(f"  Loaded from cache in {time.time()-t0:.1f}s")
    else:
        print("  Cache not found. Normalizing from raw TSVs (this will take ~25 min)...")
        print("  TIP: Run `python preprocess_data.py` once to create the cache.")
        s1_raw, s2_raw, s3_raw, _ = load_all_train(TRAIN_DIR)
        s1_full = normalize_records(s1_raw)
        s2 = normalize_records(s2_raw)
        s3 = normalize_records(s3_raw)
        print(f"  Normalization done in {time.time()-t0:.1f}s")

    # Load ground truth (always from TSV)
    from src.shared.data_loader import load_ground_truth
    gt = load_ground_truth(f"{TRAIN_DIR}/train_ground_truth.tsv")

    # Sample S1 if requested
    if sample_s1 and len(s1_full) > sample_s1:
        print(f"  Sampling {sample_s1:,} S1 entities...")
        s1 = s1_full.sample(n=sample_s1, random_state=42).reset_index(drop=True)
        sampled_ids = set(s1[COL_ENTITY_ID])
        gt_filtered = gt[gt["source1_entity_id"].isin(sampled_ids)].copy()
    else:
        s1 = s1_full
        gt_filtered = gt

    n_s23 = len(s2) + len(s3)
    print(f"  S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,}, "
          f"GT pairs={len(gt_filtered):,}")
    print(f"  Setup complete in {time.time()-t0:.1f}s")

    return s1, s2, s3, gt_filtered, n_s23


def run_experiment(
    exp_id: str,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt: pd.DataFrame,
    n_source23: int,
) -> Dict:
    """Run a single blocking experiment and return metrics."""
    exp = EXPERIMENTS[exp_id]
    print(f"\n{'='*70}")
    print(f"EXPERIMENT {exp_id}: {exp['description']}")
    print(f"{'='*70}")

    config = _make_config(exp)
    t_start = time.time()

    pairs = generate_candidates(s1, s2, s3, config)
    t_gen = time.time() - t_start

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
    print(f"  Runtime (blocking only): {t_gen:.1f}s")

    out_path = RESULTS_DIR / f"{exp_id}_results.json"
    with open(out_path, "w") as f:
        safe_metrics = {
            k: v for k, v in metrics.items()
            if k not in ("missed_pairs_sample",)
            and not (isinstance(v, float) and v != v)
        }
        json.dump(safe_metrics, f, indent=2)
    print(f"  Results: {out_path}")

    return metrics


def build_results_table(all_metrics: List[Dict]) -> pd.DataFrame:
    rows = []
    for m in all_metrics:
        rr = m.get("reduction_ratio", float("nan"))
        rows.append({
            "experiment": m["experiment_id"],
            "description": m.get("description", "")[:50],
            "recall": f"{m['blocking_recall']:.4f}",
            "true_recalled": f"{m['true_pairs_recalled']:,} / {m['total_true_pairs']:,}",
            "total_candidates": f"{m['total_candidates']:,}",
            "avg_cands": f"{m['avg_candidates_per_s1']:.1f}",
            "median_cands": f"{m['median_candidates_per_s1']:.1f}",
            "p95_cands": f"{m['p95_candidates_per_s1']:.1f}",
            "reduction_ratio": f"{rr:.6f}" if isinstance(rr, float) and rr == rr else "N/A",
            "runtime_s": f"{m['runtime_seconds']:.1f}",
        })
    return pd.DataFrame(rows)


def write_markdown_report(all_metrics: List[Dict], n_s1: int, n_s23: int) -> str:
    """Write experiment results to markdown."""
    table = build_results_table(all_metrics)
    md_path = _REPO_ROOT / "experiments" / "blocking_v1.md"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Blocking Experiment Results\n\n")
        f.write(f"**S1 entities evaluated:** {n_s1:,}  \n")
        f.write(f"**Candidate pool (S2+S3):** {n_s23:,}  \n\n")
        f.write("> Target: blocking_recall ≥ 0.98  \n\n")
        f.write("## Summary Table\n\n")
        f.write(table.to_markdown(index=False))
        f.write("\n\n## Per-Strategy Recall Breakdown\n\n")
        for m in all_metrics:
            if m.get("per_strategy_recall"):
                f.write(f"### {m['experiment_id']}: {m.get('description','')}\n\n")
                for strat, recall in sorted(m["per_strategy_recall"].items()):
                    f.write(f"- **{strat}**: {recall:.4f}\n")
                f.write("\n")
    return str(md_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N S1 entities for faster runs")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run specific experiment (e.g., B1)")
    parser.add_argument("--quick", action="store_true",
                        help="Run B0,B1,B2 only (quick sanity check)")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("BLOCKING EXPERIMENT SUITE — Member A")
    print("="*70)

    s1, s2, s3, gt, n_s23 = load_normalized_data(sample_s1=args.sample)

    if args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"ERROR: Unknown experiment '{args.experiment}'.")
            print(f"Valid: {list(EXPERIMENTS.keys())}")
            sys.exit(1)
        exp_ids = [args.experiment]
    elif args.quick:
        exp_ids = ["B0", "B1", "B2"]
    else:
        exp_ids = sorted(EXPERIMENTS.keys())

    all_metrics = []
    for exp_id in exp_ids:
        try:
            metrics = run_experiment(exp_id, s1, s2, s3, gt, n_s23)
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Experiment {exp_id} failed: {e}", exc_info=True)

    if len(all_metrics) > 1:
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        table = build_results_table(all_metrics)
        print(table.to_string(index=False))

        md_path = write_markdown_report(all_metrics, len(s1), n_s23)
        print(f"\nMarkdown report: {md_path}")


if __name__ == "__main__":
    main()
