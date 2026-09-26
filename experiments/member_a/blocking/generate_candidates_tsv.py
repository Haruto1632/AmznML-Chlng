"""
generate_candidates_tsv.py — Standalone script to generate candidate_pairs.tsv
for the full test set using the implemented blocking pipeline.

Owner: Member A | Branch: feature/blocking

Usage:
    python generate_candidates_tsv.py [--mode test|train] [--output output/]

This script:
1. Loads the test (or train) data
2. Normalizes all records
3. Runs all enabled blocking strategies
4. Consolidates candidates
5. Writes candidate_pairs.tsv in the required format

candidate_pairs.tsv format:
    source1_entity_id    candidate_entity_ids
    S1-xxx               S2-yyy,S2-zzz,S3-aaa
    (one row per S1 entity; comma-separated candidate IDs; empty if singleton)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "code" / "business_entity_resolution"))

import pandas as pd

from src.shared.data_loader import load_all_train, load_all_test
from src.shared.schemas import COL_ENTITY_ID
from src.normalization.normalizer import normalize_records
from src.blocking.blocker import generate_candidates
from src.candidate_generation.candidate_store import consolidate_candidates
from src.pipeline.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def write_candidate_pairs_tsv(
    candidate_pairs: pd.DataFrame,
    all_s1_ids: list,
    output_path: str,
) -> None:
    """
    Write candidate_pairs.tsv with required format:
        source1_entity_id    candidate_entity_ids

    Every S1 entity gets exactly one row.
    Empty candidate list → empty string in candidate_entity_ids column.
    """
    from src.shared.schemas import COL_SOURCE1_ID, COL_CANDIDATE_ID

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Group candidates per S1
    if len(candidate_pairs) > 0:
        grouped = (
            candidate_pairs
            .groupby(COL_SOURCE1_ID)[COL_CANDIDATE_ID]
            .apply(lambda ids: ",".join(dict.fromkeys(ids)))  # dedup, preserve order
            .reset_index()
            .rename(columns={COL_CANDIDATE_ID: "candidate_entity_ids"})
        )
    else:
        grouped = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_ids"])

    # Ensure every S1 entity has a row
    all_s1_df = pd.DataFrame({"source1_entity_id": all_s1_ids})
    result = all_s1_df.merge(grouped, on="source1_entity_id", how="left")
    result["candidate_entity_ids"] = result["candidate_entity_ids"].fillna("")

    result.to_csv(output_path, sep="\t", index=False)
    logger.info("Written %d rows to %s", len(result), output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate candidate_pairs.tsv")
    parser.add_argument("--mode", choices=["test", "train"], default="test")
    parser.add_argument("--output", default="output/candidate_pairs.tsv")
    parser.add_argument("--max-candidates", type=int, default=200,
                        help="Max candidates per S1 entity (default: 200)")
    args = parser.parse_args()

    config = get_config()
    config["candidate_generation"]["max_candidates_per_s1"] = args.max_candidates

    t0 = time.time()

    if args.mode == "test":
        data_dir = config["data_dir_test"]
        logger.info("Loading TEST data from: %s", data_dir)
        s1, s2, s3 = load_all_test(data_dir)
        gt = None
    else:
        data_dir = config["data_dir_train"]
        logger.info("Loading TRAIN data from: %s", data_dir)
        s1, s2, s3, gt = load_all_train(data_dir)

    logger.info("S1=%d, S2=%d, S3=%d", len(s1), len(s2), len(s3))

    logger.info("Normalizing records...")
    s1_norm = normalize_records(s1)
    s2_norm = normalize_records(s2)
    s3_norm = normalize_records(s3)
    logger.info("Normalization done in %.1fs", time.time() - t0)

    logger.info("Generating candidates...")
    raw_pairs = generate_candidates(s1_norm, s2_norm, s3_norm, config)

    logger.info("Consolidating %d raw pairs...", len(raw_pairs))
    all_s1_ids = s1[COL_ENTITY_ID].tolist()
    consolidated = consolidate_candidates(raw_pairs, all_s1_ids, config)

    logger.info("Writing candidate_pairs.tsv...")
    write_candidate_pairs_tsv(consolidated, all_s1_ids, args.output)

    logger.info("Total time: %.1fs", time.time() - t0)
    logger.info("Done. Output: %s", args.output)


if __name__ == "__main__":
    main()
