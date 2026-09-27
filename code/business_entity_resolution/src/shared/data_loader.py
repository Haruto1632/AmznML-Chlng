"""
data_loader.py — TSV loading utilities for the Amazon ML Challenge.

All teammates use these functions. Never read TSVs directly with pandas
defaults — the sep="\t" requirement is critical.

Owner: Member C  |  Branch: feature/evaluation-pipeline
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .schemas import (
    COL_ENTITY_ID,
    COL_BUSINESS_NAME,
    COL_BUSINESS_ADDRESS,
    COL_COUNTRY,
    COL_SOURCE1_ID,
    COL_MATCHED_IDS,
    SOURCE_REQUIRED_COLS,
    GT_REQUIRED_COLS,
)


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def load_source(path: str) -> pd.DataFrame:
    """
    Load one source TSV (source1, source2, or source3).

    Parameters
    ----------
    path : str
        Path to the TSV file.

    Returns
    -------
    pd.DataFrame
        Columns: entity_id, business_name, business_address, country
        All columns are str dtype; NaN values become empty strings "".

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    missing = [c for c in SOURCE_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"File {path} is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Normalise: fill NaN with empty string so downstream code never sees NaN
    for col in SOURCE_REQUIRED_COLS:
        df[col] = df[col].fillna("").str.strip()

    return df[SOURCE_REQUIRED_COLS].copy()


def load_ground_truth(path: str) -> pd.DataFrame:
    """
    Load train_ground_truth.tsv.

    Parameters
    ----------
    path : str
        Path to train_ground_truth.tsv.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, matched_entity_ids
        matched_entity_ids is a str (comma-separated IDs or empty string).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    missing = [c for c in GT_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Ground truth file {path} is missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    df[COL_SOURCE1_ID] = df[COL_SOURCE1_ID].fillna("").str.strip()
    df[COL_MATCHED_IDS] = df[COL_MATCHED_IDS].fillna("").str.strip()

    return df[[COL_SOURCE1_ID, COL_MATCHED_IDS]].copy()


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def load_all_train(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all four training files.

    Parameters
    ----------
    data_dir : str
        Directory containing train_source1.tsv, train_source2.tsv,
        train_source3.tsv, and train_ground_truth.tsv.

    Returns
    -------
    (source1, source2, source3, ground_truth) DataFrames
    """
    s1 = load_source(os.path.join(data_dir, "train_source1.tsv"))
    s2 = load_source(os.path.join(data_dir, "train_source2.tsv"))
    s3 = load_source(os.path.join(data_dir, "train_source3.tsv"))
    gt = load_ground_truth(os.path.join(data_dir, "train_ground_truth.tsv"))
    return s1, s2, s3, gt


def load_all_test(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three test files.

    Parameters
    ----------
    data_dir : str
        Directory containing test_source1.tsv, test_source2.tsv,
        test_source3.tsv.

    Returns
    -------
    (source1, source2, source3) DataFrames
    """
    s1 = load_source(os.path.join(data_dir, "test_source1.tsv"))
    s2 = load_source(os.path.join(data_dir, "test_source2.tsv"))
    s3 = load_source(os.path.join(data_dir, "test_source3.tsv"))
    return s1, s2, s3


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------

def ground_truth_to_dict(gt_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Convert ground truth DataFrame to a dict for easy lookup.

    Returns
    -------
    dict
        {source1_entity_id: [matched_entity_id, ...]}
        Singletons map to an empty list [].
    """
    result: Dict[str, List[str]] = {}
    if gt_df[COL_SOURCE1_ID].duplicated().any():
        raise ValueError("Duplicate Source 1 IDs in ground truth")
    for s1_id, matched_str in gt_df[[COL_SOURCE1_ID, COL_MATCHED_IDS]].itertuples(index=False, name=None):
        matched_str = matched_str.strip()
        if matched_str:
            result[s1_id] = [m.strip() for m in matched_str.split(",") if m.strip()]
        else:
            result[s1_id] = []
    return result


def records_to_dict(df: pd.DataFrame) -> Dict[str, dict]:
    """
    Convert a source DataFrame to a dict keyed by entity_id.

    Returns
    -------
    dict
        {entity_id: {column: value, ...}}
    """
    return df.set_index(COL_ENTITY_ID).to_dict(orient="index")


# ---------------------------------------------------------------------------
# Dataset statistics (useful for data exploration — Task A-1)
# ---------------------------------------------------------------------------

def print_dataset_stats(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt: Optional[pd.DataFrame] = None,
) -> None:
    """Print basic statistics about the loaded datasets."""
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    for name, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3)]:
        print(f"\n{name}: {len(df):,} records")
        for col in SOURCE_REQUIRED_COLS:
            n_empty = (df[col] == "").sum()
            pct = 100 * n_empty / len(df) if len(df) > 0 else 0
            print(f"  {col:30s}  empty: {n_empty:,} ({pct:.1f}%)")
        if COL_COUNTRY in df.columns:
            countries = df[COL_COUNTRY].value_counts()
            print(f"  Countries ({len(countries)} unique): {countries.head(10).to_dict()}")

    if gt is not None:
        gt_dict = ground_truth_to_dict(gt)
        total = len(gt_dict)
        singletons = sum(1 for v in gt_dict.values() if len(v) == 0)
        total_matches = sum(len(v) for v in gt_dict.values())
        print(f"\nGround Truth: {total:,} S1 entities")
        print(f"  Singletons (no match): {singletons:,} ({100*singletons/total:.1f}%)")
        print(f"  Total true match pairs: {total_matches:,}")
        if total > singletons:
            avg = total_matches / (total - singletons)
            print(f"  Avg matches per matched entity: {avg:.2f}")
    print("=" * 60)
