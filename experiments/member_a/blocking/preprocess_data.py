"""
preprocess_data.py — Pre-normalize all training data to disk (parquet cache).

Owner: Member A | Branch: feature/blocking

Run this ONCE to create normalized parquet files. All subsequent experiment
runs load the pre-processed cache instead of re-normalizing.

Estimated time (first run):
  S1 (2.2M):    ~4 min
  S2 (5.0M):    ~10 min  
  S3 (5.3M):    ~10 min

Output (in experiments/member_a/blocking/cache/):
  s1_train_norm.parquet
  s2_train_norm.parquet
  s3_train_norm.parquet

Usage:
  python preprocess_data.py
  python preprocess_data.py --output path/to/cache/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "code" / "business_entity_resolution"))

import pandas as pd

from src.shared.data_loader import load_source
from src.normalization.normalizer import normalize_records

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TRAIN_DIR = str(_REPO_ROOT / "student_resource" / "dataset" / "train")
TEST_DIR = str(_REPO_ROOT / "student_resource" / "dataset" / "test")
DEFAULT_CACHE = Path(__file__).parent / "cache"


def normalize_and_save(source_path: str, cache_path: Path, chunk_size: int = 500_000) -> None:
    """
    Load, normalize, and save a source file to parquet.
    Processes in chunks to manage memory.
    """
    if cache_path.exists():
        logger.info(f"Cache exists: {cache_path} — skipping. Delete to re-process.")
        return

    logger.info(f"Processing: {source_path} → {cache_path}")
    t0 = time.time()

    # Read full file
    df = load_source(source_path)
    logger.info(f"  Loaded {len(df):,} records in {time.time()-t0:.1f}s")

    # Normalize
    t1 = time.time()
    norm = normalize_records(df)
    logger.info(f"  Normalized in {time.time()-t1:.1f}s")

    # Convert frozensets/lists to strings for parquet serialization
    norm["business_name_tokens_str"] = norm["business_name_tokens"].apply(
        lambda s: "|".join(sorted(s)) if s else ""
    )
    norm["address_numbers_str"] = norm["address_numbers"].apply(
        lambda lst: "|".join(lst) if lst else ""
    )

    # Drop non-serializable columns (frozenset, list)
    cols_to_drop = ["business_name_tokens", "address_numbers"]
    norm_serializable = norm.drop(columns=cols_to_drop, errors="ignore")

    # Save to parquet
    t2 = time.time()
    norm_serializable.to_parquet(cache_path, index=False)
    logger.info(f"  Saved to {cache_path} in {time.time()-t2:.1f}s")
    logger.info(f"  Total: {time.time()-t0:.1f}s")


def load_normalized(cache_path: Path) -> pd.DataFrame:
    """
    Load pre-normalized data from parquet and reconstruct Python types.
    """
    df = pd.read_parquet(cache_path)

    # Reconstruct frozenset from string representation
    df["business_name_tokens"] = df["business_name_tokens_str"].apply(
        lambda s: frozenset(s.split("|")) if s else frozenset()
    )
    df["address_numbers"] = df["address_numbers_str"].apply(
        lambda s: s.split("|") if s else []
    )

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_CACHE))
    parser.add_argument("--test", action="store_true", help="Also process test set")
    args = parser.parse_args()

    cache_dir = Path(args.output)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCache directory: {cache_dir}")
    print("="*60)

    t_total = time.time()

    normalize_and_save(
        f"{TRAIN_DIR}/train_source1.tsv",
        cache_dir / "s1_train_norm.parquet",
    )
    normalize_and_save(
        f"{TRAIN_DIR}/train_source2.tsv",
        cache_dir / "s2_train_norm.parquet",
    )
    normalize_and_save(
        f"{TRAIN_DIR}/train_source3.tsv",
        cache_dir / "s3_train_norm.parquet",
    )

    if args.test:
        normalize_and_save(
            f"{TEST_DIR}/test_source1.tsv",
            cache_dir / "s1_test_norm.parquet",
        )
        normalize_and_save(
            f"{TEST_DIR}/test_source2.tsv",
            cache_dir / "s2_test_norm.parquet",
        )
        normalize_and_save(
            f"{TEST_DIR}/test_source3.tsv",
            cache_dir / "s3_test_norm.parquet",
        )

    print(f"\nTotal preprocessing time: {(time.time()-t_total)/60:.1f} min")
    print(f"Cache ready at: {cache_dir}")


if __name__ == "__main__":
    main()
