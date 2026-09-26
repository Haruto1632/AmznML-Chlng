"""
hard_negatives.py — Hard negative mining for training data.

Owner: Member B  |  Branch: feature/matching-model

Hard negatives are non-matching pairs that look superficially similar.
They prevent the model from relying on trivially easy negatives.

TODO (Member B): Implement mine_hard_negatives().
See TEAM_TASKS.md Task B-2.
"""

from __future__ import annotations

from typing import Dict, List, Set

import pandas as pd


def mine_hard_negatives(
    ground_truth_dict: Dict[str, List[str]],
    candidate_pairs: pd.DataFrame,
    feature_df: pd.DataFrame,
    n_per_positive: int = 3,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Mine hard negative pairs from the candidate set.

    Hard negatives are candidate pairs that:
    - Are NOT in the ground truth (i.e., not true matches)
    - Have high feature similarity (confusing for the model)

    The best source of hard negatives is the blocking output itself:
    any pair that the blocking step found but that is NOT a true match.

    Parameters
    ----------
    ground_truth_dict : dict
        {source1_entity_id: [true_matched_ids]}
    candidate_pairs : pd.DataFrame
        All blocking-generated candidate pairs.
    feature_df : pd.DataFrame
        Precomputed features for all candidate pairs.
    n_per_positive : int
        Number of hard negatives to mine per positive pair.
    random_seed : int

    Returns
    -------
    pd.DataFrame
        Subset of feature_df rows that are hard negatives (label=0).
    """
    # TODO: implement
    raise NotImplementedError(
        "mine_hard_negatives() not yet implemented. See TEAM_TASKS.md Task B-2."
    )
