"""
config.py — Central configuration for the pipeline.

All teammates reference config keys from here.
Owner: Member C  |  Branch: feature/evaluation-pipeline
"""

from __future__ import annotations

import os
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    # -----------------------------------------------------------------------
    # Paths (relative to workspace root; student_resource/ contains the data)
    # -----------------------------------------------------------------------
    "data_dir_train": "student_resource/dataset/train",
    "data_dir_test": "student_resource/dataset/test",
    "output_dir": "output",
    "model_dir": "code/business_entity_resolution/models",

    # -----------------------------------------------------------------------
    # Normalization
    # -----------------------------------------------------------------------
    "normalization": {
        "expand_legal_suffixes": True,
        "normalize_ampersand": True,     # & → and
        "sort_name_tokens": True,        # create token-sorted column
        "transliterate": True,           # local only, no API
        "address_expand_abbreviations": True,
    },

    # -----------------------------------------------------------------------
    # Blocking
    # -----------------------------------------------------------------------
    "blocking": {
        "strategy_exact_token": True,
        "strategy_token_overlap": True,
        "strategy_ngram_lsh": True,
        "strategy_phonetic": True,
        "strategy_address_token": True,
        "strategy_embedding_ann": False,   # disabled until embedding model confirmed
        # Strategy-specific thresholds
        "token_overlap_threshold": 0.3,
        "ngram_lsh_num_perm": 128,
        "ngram_lsh_threshold": 0.25,
        "ngram_n": 3,
    },

    # -----------------------------------------------------------------------
    # Candidate consolidation
    # -----------------------------------------------------------------------
    "candidate_generation": {
        "max_candidates_per_s1": 50,     # hard cap per S1 entity — SCALE MATTERS (2.2M S1 entities)
        "prefer_high_evidence": True,    # when trimming, keep pairs with more strategies
    },

    # -----------------------------------------------------------------------
    # Features
    # -----------------------------------------------------------------------
    "features": {
        "use_name_features": True,
        "use_address_features": True,
        "use_country_features": True,
        "use_structural_features": True,
        "use_embedding_features": False,  # disabled until model confirmed
        "tfidf_char_ngram_n": 3,
        "tfidf_max_features": 50000,
    },

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    "model": {
        "type": "gbm",                   # "gbm" | "baseline_rule" | "ensemble"
        "gbm_backend": "auto",           # "lightgbm" | "xgboost" | "histgbm" | "auto"
        "random_seed": 42,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "class_weight": "balanced",
        "early_stopping_rounds": 50,
        # Training data
        "negative_sample_ratio": 5,      # negatives per positive
        "hard_negative_fraction": 0.6,   # fraction of negatives that are hard negatives
        "val_fraction": 0.2,
        "stratify_by_singleton": True,
    },

    # -----------------------------------------------------------------------
    # Calibration
    # -----------------------------------------------------------------------
    "calibration": {
        "thresholds": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        "metric": "f05",                 # calibrate on F_0.5
        "default_threshold": 0.70,       # used if no validation data available
    },

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    "output": {
        "matching_results_filename": "matching_results.tsv",
        "candidate_pairs_filename": "candidate_pairs.tsv",
    },
}


def get_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Return a copy of DEFAULT_CONFIG, optionally updated with overrides.

    Supports nested overrides via dot notation or nested dict.
    """
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)
    if overrides:
        _deep_update(config, overrides)
    return config


def _deep_update(base: dict, overrides: dict) -> None:
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
