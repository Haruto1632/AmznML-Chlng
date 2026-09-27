"""
blocker.py — Multi-strategy blocking orchestrator.

Owner: Member A  |  Branch: feature/blocking

Entry point used by the shared pipeline:
    from ..blocking.blocker import generate_candidates

Orchestrates all blocking strategies, unions results, tracks provenance.

Interface contract (from TEAM_TASKS.md Task A-3):
    generate_candidates(source1, source2, source3, config)
        -> DataFrame[source1_entity_id, candidate_entity_id, blocking_reasons]

All source DataFrames must already be normalized (normalizer.normalize_records called).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Set

import pandas as pd

from ..shared.schemas import (
    COL_ENTITY_ID,
    COL_SOURCE1_ID,
    COL_CANDIDATE_ID,
    COL_BLOCKING_REASONS,
    STRATEGY_EXACT_TOKEN,
    STRATEGY_TOKEN_OVERLAP,
    STRATEGY_NGRAM_LSH,
    STRATEGY_PHONETIC,
    STRATEGY_ADDRESS_TOKEN,
    STRATEGY_EMBEDDING_ANN,
)
from .strategies.token_blocking import (
    _build_token_index,
    generate_exact_country_token_pairs,
    generate_token_overlap_pairs,
)
from .strategies.phonetic_blocking import (
    build_phonetic_index,
    generate_phonetic_pairs,
)
from .strategies.ngram_blocking import (
    build_ngram_lsh,
    generate_ngram_lsh_pairs,
)
from .strategies.address_blocking import (
    build_address_index,
    generate_address_pairs,
)
from .strategies.ann_blocking import generate_ann_pairs, ann_available

logger = logging.getLogger(__name__)


def generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Generate candidate pairs using all enabled blocking strategies.

    Parameters
    ----------
    source1 : pd.DataFrame
        Normalized S1 records (from normalizer.normalize_records).
    source2 : pd.DataFrame
        Normalized S2 records.
    source3 : pd.DataFrame
        Normalized S3 records.
    config : dict
        Pipeline config. See src/pipeline/config.py DEFAULT_CONFIG.
        Relevant keys under config["blocking"]:
            strategy_exact_token     (bool)
            strategy_token_overlap   (bool)
            strategy_ngram_lsh       (bool)
            strategy_phonetic        (bool)
            strategy_address_token   (bool)
            strategy_embedding_ann   (bool)
            token_overlap_threshold  (float, unused for now)
            ngram_lsh_num_perm       (int)
            ngram_lsh_threshold      (float)
            ngram_n                  (int)
            token_max_df_frac        (float, default 0.01)

    Returns
    -------
    pd.DataFrame
        Columns:
          source1_entity_id   (str)
          candidate_entity_id (str)
          blocking_reasons    (list[str])  — strategies that produced this pair
        One row per unique (source1_entity_id, candidate_entity_id) pair.
        Every pair appears at most once; blocking_reasons is the union of all
        strategies that found it.
    """
    t_start = time.time()
    bcfg = config.get("blocking", {})

    # ── Combine S2 + S3 into one candidate pool ───────────────────────────
    logger.info("Combining S2 (%d) + S3 (%d) into candidate pool...",
                len(source2), len(source3))
    candidates = pd.concat([source2, source3], ignore_index=True)
    logger.info("Candidate pool: %d records", len(candidates))

    # ── Strategy results accumulator ─────────────────────────────────────
    # Maps (s1_id, cand_id) → list of strategy names
    pair_reasons: Dict[tuple, List[str]] = defaultdict(list)

    def _accumulate(pairs_df: pd.DataFrame, strategy_name: str) -> None:
        """Add strategy pairs to the global accumulator."""
        if pairs_df is None or len(pairs_df) == 0:
            logger.info("  %s: 0 pairs", strategy_name)
            return
        n = len(pairs_df)
        logger.info("  %s: %d pairs", strategy_name, n)
        for row in pairs_df.itertuples(index=False):
            pair_reasons[(row.source1_entity_id, row.candidate_entity_id)].append(
                strategy_name
            )

    # ── Pre-build shared indices ──────────────────────────────────────────
    max_df_frac = bcfg.get("token_max_df_frac", 0.001)   # 0.1% default
    max_bucket_size = bcfg.get("token_max_bucket_size", 5000)  # cap per bucket

    # Token indices (shared by both token strategies)
    needs_token = (
        bcfg.get("strategy_exact_token", True)
        or bcfg.get("strategy_token_overlap", True)
    )
    country_token_index = token_index = stop_tokens = None
    if needs_token:
        logger.info("Building token inverted index...")
        t0 = time.time()
        country_token_index, token_index, stop_tokens = _build_token_index(
            candidates,
            max_df_frac=max_df_frac,
            max_bucket_size=max_bucket_size,
        )
        logger.info("  Token index built in %.1fs", time.time() - t0)

    # Phonetic index
    phonetic_index = None
    if bcfg.get("strategy_phonetic", True):
        logger.info("Building phonetic index...")
        t0 = time.time()
        phonetic_index = build_phonetic_index(candidates)
        logger.info("  Phonetic index built in %.1fs", time.time() - t0)

    # Address index
    number_index = number_name_index = None
    if bcfg.get("strategy_address_token", True):
        logger.info("Building address index...")
        t0 = time.time()
        number_index, number_name_index = build_address_index(candidates)
        logger.info("  Address index built in %.1fs", time.time() - t0)

    # N-gram LSH index
    lsh = None
    ngram_num_perm = bcfg.get("ngram_lsh_num_perm", 64)
    ngram_threshold = bcfg.get("ngram_lsh_threshold", 0.25)
    ngram_n = bcfg.get("ngram_n", 3)
    if bcfg.get("strategy_ngram_lsh", True):
        logger.info("Building MinHash LSH index (num_perm=%d, threshold=%.2f)...",
                    ngram_num_perm, ngram_threshold)
        t0 = time.time()
        lsh, _ = build_ngram_lsh(
            candidates,
            num_perm=ngram_num_perm,
            threshold=ngram_threshold,
            ngram_n=ngram_n,
        )
        logger.info("  LSH index built in %.1fs", time.time() - t0)

    # ── Run strategies ────────────────────────────────────────────────────
    logger.info("\nRunning blocking strategies on %d S1 entities...", len(source1))

    max_cands = bcfg.get("max_strategy_candidates", 200)

    if bcfg.get("strategy_exact_token", True) and country_token_index is not None:
        logger.info("Strategy 1: %s", STRATEGY_EXACT_TOKEN)
        pairs = generate_exact_country_token_pairs(
            source1, candidates, country_token_index, stop_tokens
        )
        _accumulate(pairs, STRATEGY_EXACT_TOKEN)

    if bcfg.get("strategy_token_overlap", True) and token_index is not None:
        logger.info("Strategy 2: %s", STRATEGY_TOKEN_OVERLAP)
        pairs = generate_token_overlap_pairs(
            source1, candidates, token_index, stop_tokens
        )
        _accumulate(pairs, STRATEGY_TOKEN_OVERLAP)

    if bcfg.get("strategy_phonetic", True) and phonetic_index is not None:
        logger.info("Strategy 3: %s", STRATEGY_PHONETIC)
        pairs = generate_phonetic_pairs(source1, phonetic_index)
        _accumulate(pairs, STRATEGY_PHONETIC)

    if bcfg.get("strategy_address_token", True) and number_index is not None:
        logger.info("Strategy 4: %s", STRATEGY_ADDRESS_TOKEN)
        pairs = generate_address_pairs(
            source1, number_index, number_name_index,
            use_joint_index=True
        )
        _accumulate(pairs, STRATEGY_ADDRESS_TOKEN)

    if bcfg.get("strategy_ngram_lsh", True) and lsh is not None:
        logger.info("Strategy 5: %s", STRATEGY_NGRAM_LSH)
        pairs = generate_ngram_lsh_pairs(
            source1, lsh,
            num_perm=ngram_num_perm,
            ngram_n=ngram_n,
        )
        _accumulate(pairs, STRATEGY_NGRAM_LSH)

    if bcfg.get("strategy_embedding_ann", False):
        logger.info("Strategy 6: %s", STRATEGY_EMBEDDING_ANN)
        pairs = generate_ann_pairs(source1, candidates, config)
        _accumulate(pairs, STRATEGY_EMBEDDING_ANN)

    # ── Convert accumulator to DataFrame ─────────────────────────────────
    logger.info("\nConverting %d unique pairs to DataFrame...", len(pair_reasons))

    rows = [
        {
            COL_SOURCE1_ID: s1_id,
            COL_CANDIDATE_ID: cand_id,
            COL_BLOCKING_REASONS: reasons,
        }
        for (s1_id, cand_id), reasons in pair_reasons.items()
    ]

    if not rows:
        logger.warning("No candidate pairs generated!")
        result = pd.DataFrame(columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID, COL_BLOCKING_REASONS])
    else:
        result = pd.DataFrame(rows)

    t_total = time.time() - t_start
    logger.info(
        "Blocking complete: %d unique pairs | %.1fs total",
        len(result), t_total
    )

    return result
