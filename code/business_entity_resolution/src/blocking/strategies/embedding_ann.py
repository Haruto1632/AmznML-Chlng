"""
embedding_ann.py — ANN blocking on pretrained sentence embeddings.

Strategy: encode business names with a pretrained sentence transformer,
then use FAISS or a brute-force ANN search to find nearest neighbours.

Owner: Member A  |  Branch: feature/blocking

REQUIREMENTS:
  - Controlled by config["blocking"]["strategy_embedding_ann"] = True/False
  - Disabled by default (requires sentence-transformers + torch)
  - Only MIT / Apache 2.0 licensed models allowed
  - Model parameters <= 8B

Suggested models (Apache 2.0 / MIT):
  - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  (multilingual, 117M params)
  - "sentence-transformers/all-MiniLM-L6-v2"                       (English only, 22M params)
"""

from __future__ import annotations

import pandas as pd

from ...shared.schemas import STRATEGY_EMBEDDING_ANN


def run(
    source1: pd.DataFrame,
    candidates: pd.DataFrame,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns (source1_entity_id, candidate_entity_id).

    Note: This strategy is computationally expensive. Use only if other
    strategies miss too many true matches.
    """
    # TODO: implement (optional)
    # Approach:
    #   1. Load pretrained sentence transformer
    #   2. Encode all S2/S3 business names
    #   3. For each S1 entity, find top_k nearest neighbours
    #   4. Return as candidate pairs
    raise NotImplementedError("embedding_ann.run() not yet implemented (optional strategy).")
