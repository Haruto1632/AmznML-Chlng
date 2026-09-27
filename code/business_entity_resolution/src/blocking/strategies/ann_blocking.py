"""
ann_blocking.py — Embedding-based ANN (Approximate Nearest Neighbor) blocking.

Owner: Member A  |  Branch: feature/blocking

Strategy: embedding_ann
  Status: DISABLED — faiss and sentence_transformers not installed.

This module is a clean stub that reports unavailability gracefully.
When faiss and sentence_transformers are available, this module would:
  1. Load a small MIT/Apache-compatible sentence-transformer model
  2. Batch-encode all S2+S3 business name + address strings
  3. Build a FAISS index (IVFFlat or HNSWFlat)
  4. Query with each S1 entity to find top-k approximate neighbors
  5. Return pairs as DataFrame

To enable: pip install faiss-cpu sentence-transformers
"""

from __future__ import annotations

import logging

import pandas as pd

from ...shared.schemas import COL_SOURCE1_ID, COL_CANDIDATE_ID

logger = logging.getLogger(__name__)

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

_ANN_AVAILABLE = _HAS_FAISS and _HAS_SENTENCE_TRANSFORMERS


def generate_ann_pairs(
    source1: pd.DataFrame,
    candidates: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Strategy: embedding_ann (ANN over sentence embeddings).

    Returns empty DataFrame with correct columns if dependencies unavailable.
    """
    if not _ANN_AVAILABLE:
        logger.info(
            "ANN blocking SKIPPED: faiss=%s, sentence_transformers=%s. "
            "Install both packages to enable.",
            _HAS_FAISS,
            _HAS_SENTENCE_TRANSFORMERS,
        )
        return pd.DataFrame(columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])

    # Full implementation would go here when dependencies are available.
    # Not implemented in this version.
    logger.info("ANN blocking: dependencies available but not yet implemented.")
    return pd.DataFrame(columns=[COL_SOURCE1_ID, COL_CANDIDATE_ID])


def ann_available() -> bool:
    """Return True if ANN blocking dependencies are installed."""
    return _ANN_AVAILABLE
