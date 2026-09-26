"""
output_writer.py — Write and validate output TSV files.

Owner: Member C  |  Branch: feature/evaluation-pipeline

Implements all output format guarantees from SUBMISSION_CHECKLIST.md.
"""

from __future__ import annotations

import os
from typing import Dict, List, Set

import pandas as pd


def _ids_to_str(ids: List[str]) -> str:
    """Convert a list of IDs to a comma-separated string. Empty list → empty string."""
    return ",".join(ids) if ids else ""


def write_matching_results(
    predictions: Dict[str, List[str]],
    all_s1_ids: List[str],
    path: str,
) -> None:
    """
    Write matching_results.tsv.

    Parameters
    ----------
    predictions : dict
        {source1_entity_id: [matched_entity_ids]}
    all_s1_ids : list[str]
        Every Source 1 entity ID — guarantees full coverage.
    path : str
        Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    seen = set()
    for s1_id in all_s1_ids:
        if s1_id in seen:
            raise ValueError(f"Duplicate source1_entity_id: {s1_id}")
        seen.add(s1_id)
        matched = predictions.get(s1_id, [])
        # Deduplicate while preserving order
        deduped = list(dict.fromkeys(matched))
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": _ids_to_str(deduped),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)


def write_candidate_pairs(
    candidate_pairs: pd.DataFrame,
    all_s1_ids: List[str],
    path: str,
) -> None:
    """
    Write candidate_pairs.tsv.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id (plus optional blocking_reasons)
    all_s1_ids : list[str]
        Every Source 1 entity ID.
    path : str
        Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Group candidates per S1 entity
    if len(candidate_pairs) > 0:
        grouped = (
            candidate_pairs
            .groupby("source1_entity_id")["candidate_entity_id"]
            .apply(lambda ids: _ids_to_str(list(dict.fromkeys(ids))))
            .reset_index()
            .rename(columns={"candidate_entity_id": "candidate_entity_ids"})
        )
    else:
        grouped = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_ids"])

    # Ensure every S1 entity has a row
    all_s1_df = pd.DataFrame({"source1_entity_id": all_s1_ids})
    result = all_s1_df.merge(grouped, on="source1_entity_id", how="left")
    result["candidate_entity_ids"] = result["candidate_entity_ids"].fillna("")

    result.to_csv(path, sep="\t", index=False)


def validate_output_integrity(
    matching_path: str,
    candidate_path: str,
    s1_ids: Set[str],
    s2_ids: Set[str],
    s3_ids: Set[str],
) -> List[str]:
    """
    Check output files for all submission rules.

    Returns
    -------
    list[str]
        List of violation messages. Empty list means all checks passed.
    """
    violations = []
    valid_candidate_ids = s2_ids | s3_ids

    # --- Load files ---
    try:
        matching = pd.read_csv(matching_path, sep="\t", dtype=str).fillna("")
    except Exception as e:
        return [f"Could not read matching_results.tsv: {e}"]

    try:
        candidates = pd.read_csv(candidate_path, sep="\t", dtype=str).fillna("")
    except Exception as e:
        return [f"Could not read candidate_pairs.tsv: {e}"]

    # --- matching_results.tsv checks ---
    m_s1_ids = set(matching["source1_entity_id"].tolist())
    if m_s1_ids != s1_ids:
        missing = s1_ids - m_s1_ids
        extra = m_s1_ids - s1_ids
        if missing:
            violations.append(f"matching_results.tsv: missing {len(missing)} S1 entities")
        if extra:
            violations.append(f"matching_results.tsv: {len(extra)} extra S1 entities not in test")

    if matching["source1_entity_id"].duplicated().any():
        violations.append("matching_results.tsv: duplicate source1_entity_id rows")

    # Build matched-to-candidate index for cross-check
    matched_index: Dict[str, Set[str]] = {}
    for _, row in matching.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"].strip()
        if not matched_str:
            matched_index[s1_id] = set()
            continue
        ids = [x.strip() for x in matched_str.split(",") if x.strip()]
        id_set = set()
        for mid in ids:
            if mid.startswith("S1-"):
                violations.append(
                    f"matching_results.tsv: S1 entity {mid} appears in matches for {s1_id}"
                )
            if mid not in valid_candidate_ids:
                violations.append(
                    f"matching_results.tsv: {mid} does not exist in test S2/S3"
                )
            if mid in id_set:
                violations.append(
                    f"matching_results.tsv: duplicate ID {mid} in row for {s1_id}"
                )
            id_set.add(mid)
        matched_index[s1_id] = id_set

    # --- candidate_pairs.tsv checks ---
    c_s1_ids = set(candidates["source1_entity_id"].tolist())
    if c_s1_ids != s1_ids:
        missing = s1_ids - c_s1_ids
        if missing:
            violations.append(
                f"candidate_pairs.tsv: missing {len(missing)} S1 entities"
            )

    candidate_index: Dict[str, Set[str]] = {}
    for _, row in candidates.iterrows():
        s1_id = row["source1_entity_id"]
        cand_str = row["candidate_entity_ids"].strip()
        if not cand_str:
            candidate_index[s1_id] = set()
            continue
        ids = [x.strip() for x in cand_str.split(",") if x.strip()]
        id_set = set()
        for cid in ids:
            if cid not in valid_candidate_ids:
                violations.append(
                    f"candidate_pairs.tsv: {cid} does not exist in test S2/S3"
                )
            if cid in id_set:
                violations.append(
                    f"candidate_pairs.tsv: duplicate ID {cid} in row for {s1_id}"
                )
            id_set.add(cid)
        candidate_index[s1_id] = id_set

    # --- Cross-check: every matched ID must be in candidates ---
    for s1_id, matched_ids in matched_index.items():
        candidate_ids = candidate_index.get(s1_id, set())
        not_in_candidates = matched_ids - candidate_ids
        if not_in_candidates:
            violations.append(
                f"Matched IDs not in candidate set for {s1_id}: {not_in_candidates}"
            )

    return violations
