"""
schemas.py — Shared data contracts for the Amazon ML Challenge pipeline.

ALL teammates import from here. Do not redefine these types locally.
Owner: Member C  |  Branch: feature/evaluation-pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Column name constants (single source of truth)
# ---------------------------------------------------------------------------

# Raw source file columns
COL_ENTITY_ID = "entity_id"
COL_BUSINESS_NAME = "business_name"
COL_BUSINESS_ADDRESS = "business_address"
COL_COUNTRY = "country"

# Normalized columns (added by normalization step, originals preserved)
COL_NAME_NORMALIZED = "business_name_normalized"
COL_NAME_TOKENS = "business_name_tokens"          # frozenset of tokens
COL_NAME_TOKEN_SORTED = "business_name_token_sorted"  # tokens sorted, joined
COL_ADDRESS_NORMALIZED = "business_address_normalized"
COL_ADDRESS_NUMBERS = "address_numbers"           # list of numeric tokens
COL_COUNTRY_NORMALIZED = "country_normalized"

# Ground truth columns
COL_SOURCE1_ID = "source1_entity_id"
COL_MATCHED_IDS = "matched_entity_ids"

# Candidate pair columns
COL_CANDIDATE_ID = "candidate_entity_id"
COL_BLOCKING_REASONS = "blocking_reasons"

# Output columns
COL_CANDIDATE_IDS = "candidate_entity_ids"

# Feature / prediction columns
COL_MATCH_PROB = "match_probability"
COL_LABEL = "label"

# Source prefixes
PREFIX_S1 = "S1-"
PREFIX_S2 = "S2-"
PREFIX_S3 = "S3-"

# Required columns in source TSVs
SOURCE_REQUIRED_COLS = [COL_ENTITY_ID, COL_BUSINESS_NAME, COL_BUSINESS_ADDRESS, COL_COUNTRY]

# Required columns in ground truth TSV
GT_REQUIRED_COLS = [COL_SOURCE1_ID, COL_MATCHED_IDS]


# ---------------------------------------------------------------------------
# Blocking strategy name constants
# ---------------------------------------------------------------------------

STRATEGY_EXACT_TOKEN = "exact_country_name_token"
STRATEGY_TOKEN_OVERLAP = "name_token_overlap"
STRATEGY_NGRAM_LSH = "char_ngram_lsh"
STRATEGY_PHONETIC = "phonetic"
STRATEGY_ADDRESS_TOKEN = "address_token_overlap"
STRATEGY_EMBEDDING_ANN = "embedding_ann"

ALL_STRATEGIES = [
    STRATEGY_EXACT_TOKEN,
    STRATEGY_TOKEN_OVERLAP,
    STRATEGY_NGRAM_LSH,
    STRATEGY_PHONETIC,
    STRATEGY_ADDRESS_TOKEN,
    STRATEGY_EMBEDDING_ANN,
]


# ---------------------------------------------------------------------------
# Dataclasses (used for documentation and type hints; DataFrames used in practice)
# ---------------------------------------------------------------------------

@dataclass
class NormalizedRecord:
    """One record after normalization. Originals are always preserved."""
    entity_id: str
    source: str                        # "S1", "S2", or "S3"
    business_name: str                 # original
    business_address: str              # original
    country: str                       # original
    business_name_normalized: str
    business_name_tokens: frozenset    # set of lowercase tokens
    business_name_token_sorted: str    # tokens sorted alphabetically, joined
    business_address_normalized: str
    address_numbers: List[str]         # extracted numeric tokens
    country_normalized: str


@dataclass
class CandidatePair:
    """
    A candidate pair produced by the blocking step.
    blocking_reasons lists every strategy that produced this pair.
    """
    source1_entity_id: str
    candidate_entity_id: str
    blocking_reasons: List[str] = field(default_factory=list)


@dataclass
class FeatureRow:
    """
    One row of the feature matrix for the pairwise matching model.
    The list of float feature columns is open-ended; see feature_builder.py
    for the canonical feature names.
    """
    source1_entity_id: str
    candidate_entity_id: str
    # Feature values are added as additional attributes or DataFrame columns.
    # Use FEATURE_COLUMNS list below for the canonical ordered list.


@dataclass
class Prediction:
    """Match probability output from the model scorer."""
    source1_entity_id: str
    candidate_entity_id: str
    match_probability: float


# ---------------------------------------------------------------------------
# Canonical feature column list (agreed between B and C)
# ---------------------------------------------------------------------------

NAME_FEATURE_COLUMNS = [
    "name_exact_normalized",
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_jaccard",
    "name_token_overlap_ratio",
    "name_char3gram_cosine",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_length_diff",
    "name_abbrev_score",
    "name_sorted_exact",
]

ADDRESS_FEATURE_COLUMNS = [
    "addr_exact_normalized",
    "addr_token_jaccard",
    "addr_char3gram_cosine",
    "addr_edit_similarity",
    "addr_numeric_overlap",
    "addr_length_diff",
    "addr_both_empty",
    "addr_one_empty",
]

COUNTRY_FEATURE_COLUMNS = [
    "country_exact_normalized",
    "country_both_missing",
]

STRUCTURAL_FEATURE_COLUMNS = [
    "num_blocking_strategies",
    f"strategy_{STRATEGY_EXACT_TOKEN}",
    f"strategy_{STRATEGY_TOKEN_OVERLAP}",
    f"strategy_{STRATEGY_NGRAM_LSH}",
    f"strategy_{STRATEGY_PHONETIC}",
    f"strategy_{STRATEGY_ADDRESS_TOKEN}",
    f"strategy_{STRATEGY_EMBEDDING_ANN}",
]

ALL_FEATURE_COLUMNS = (
    NAME_FEATURE_COLUMNS
    + ADDRESS_FEATURE_COLUMNS
    + COUNTRY_FEATURE_COLUMNS
    + STRUCTURAL_FEATURE_COLUMNS
)
