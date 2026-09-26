"""
test_output_writer.py — Tests for output format and integrity checks.

Owner: Member C
Run with: python -m pytest tests/test_output_writer.py -v
"""

import os
import tempfile
import pytest
import pandas as pd

from ..src.pipeline.output_writer import (
    write_matching_results,
    write_candidate_pairs,
    validate_output_integrity,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestWriteMatchingResults:
    def test_basic_write(self, tmp_dir):
        preds = {
            "S1-00001": ["S2-00001", "S3-00001"],
            "S1-00002": [],
        }
        path = os.path.join(tmp_dir, "output", "matching_results.tsv")
        write_matching_results(preds, ["S1-00001", "S1-00002"], path)
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        assert len(df) == 2
        assert set(df["source1_entity_id"]) == {"S1-00001", "S1-00002"}
        row1 = df.loc[df["source1_entity_id"] == "S1-00001", "matched_entity_ids"].iloc[0]
        assert "S2-00001" in row1
        assert "S3-00001" in row1
        row2 = df.loc[df["source1_entity_id"] == "S1-00002", "matched_entity_ids"].iloc[0]
        assert row2 == ""

    def test_all_s1_ids_appear(self, tmp_dir):
        """Entities with no predictions still appear in output."""
        preds = {}
        path = os.path.join(tmp_dir, "output", "matching_results.tsv")
        write_matching_results(preds, ["S1-00001", "S1-00002", "S1-00003"], path)
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        assert len(df) == 3

    def test_no_duplicates_in_output(self, tmp_dir):
        """Duplicate IDs in predictions are deduplicated."""
        preds = {"S1-00001": ["S2-00001", "S2-00001"]}
        path = os.path.join(tmp_dir, "output", "matching_results.tsv")
        write_matching_results(preds, ["S1-00001"], path)
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        ids = df.loc[df["source1_entity_id"] == "S1-00001", "matched_entity_ids"].iloc[0].split(",")
        assert ids.count("S2-00001") == 1

    def test_tab_separator(self, tmp_dir):
        preds = {"S1-00001": ["S2-00001"]}
        path = os.path.join(tmp_dir, "output", "matching_results.tsv")
        write_matching_results(preds, ["S1-00001"], path)
        with open(path) as f:
            header = f.readline()
        assert "\t" in header


class TestValidateOutputIntegrity:
    def _write_files(self, tmp_dir, matching_rows, candidate_rows):
        m_path = os.path.join(tmp_dir, "matching_results.tsv")
        c_path = os.path.join(tmp_dir, "candidate_pairs.tsv")
        pd.DataFrame(matching_rows).to_csv(m_path, sep="\t", index=False)
        pd.DataFrame(candidate_rows).to_csv(c_path, sep="\t", index=False)
        return m_path, c_path

    def test_clean_output_no_violations(self, tmp_dir):
        m_rows = [{"source1_entity_id": "S1-00001", "matched_entity_ids": "S2-00001"}]
        c_rows = [{"source1_entity_id": "S1-00001", "candidate_entity_ids": "S2-00001,S2-00002"}]
        m_path, c_path = self._write_files(tmp_dir, m_rows, c_rows)
        violations = validate_output_integrity(
            m_path, c_path,
            s1_ids={"S1-00001"},
            s2_ids={"S2-00001", "S2-00002"},
            s3_ids=set(),
        )
        assert violations == [], f"Unexpected violations: {violations}"

    def test_matched_id_not_in_candidates_triggers_violation(self, tmp_dir):
        m_rows = [{"source1_entity_id": "S1-00001", "matched_entity_ids": "S2-00099"}]
        c_rows = [{"source1_entity_id": "S1-00001", "candidate_entity_ids": "S2-00001"}]
        m_path, c_path = self._write_files(tmp_dir, m_rows, c_rows)
        violations = validate_output_integrity(
            m_path, c_path,
            s1_ids={"S1-00001"},
            s2_ids={"S2-00001", "S2-00099"},
            s3_ids=set(),
        )
        # S2-00099 is in matched but NOT in candidates → violation
        assert any("S2-00099" in v for v in violations)

    def test_s1_id_in_matched_ids_triggers_violation(self, tmp_dir):
        m_rows = [{"source1_entity_id": "S1-00001", "matched_entity_ids": "S1-00002"}]
        c_rows = [{"source1_entity_id": "S1-00001", "candidate_entity_ids": "S1-00002"}]
        m_path, c_path = self._write_files(tmp_dir, m_rows, c_rows)
        violations = validate_output_integrity(
            m_path, c_path,
            s1_ids={"S1-00001"},
            s2_ids=set(),
            s3_ids=set(),
        )
        assert any("S1-" in v for v in violations)

    def test_missing_s1_entity_triggers_violation(self, tmp_dir):
        m_rows = [{"source1_entity_id": "S1-00001", "matched_entity_ids": ""}]
        c_rows = [{"source1_entity_id": "S1-00001", "candidate_entity_ids": ""}]
        m_path, c_path = self._write_files(tmp_dir, m_rows, c_rows)
        violations = validate_output_integrity(
            m_path, c_path,
            s1_ids={"S1-00001", "S1-00002"},  # S1-00002 missing
            s2_ids=set(),
            s3_ids=set(),
        )
        assert any("missing" in v.lower() for v in violations)
