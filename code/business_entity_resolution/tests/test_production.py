"""Bounded index, batched features and persisted inference invariants."""
import numpy as np
import pandas as pd
import pyarrow as pa

from ..src.shared.disk_store import SCHEMA, RecordStore
from ..src.blocking.disk_index import build_index, DiskBlocker
from ..src.features.batch_features import BatchFeatures


def store(tmp_path, rows):
    path = tmp_path / "records.arrow"
    frame = pd.DataFrame(rows, columns=SCHEMA.names)
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, SCHEMA) as writer:
        for start in range(0, len(frame), 2):
            writer.write_table(pa.Table.from_pandas(frame.iloc[start:start+2], schema=SCHEMA, preserve_index=False))
    return RecordStore([path]), frame


def test_store_random_access_across_batches_and_duplicates(tmp_path):
    pool, frame = store(tmp_path, [(f"S2-{i}", "acme", "12 main", "france", "acme", "12") for i in range(7)])
    got = pool.take([6, 0, 6, 3])
    assert got.entity_id.tolist() == ["S2-6", "S2-0", "S2-6", "S2-3"]


def test_oversized_buckets_removed_before_query_and_empty_pairs(tmp_path):
    pool, frame = store(tmp_path, [(f"S2-{i}", "acme", "12 main", "france", "acme", "12") for i in range(7)])
    build_index(pool, tmp_path / "index", bucket_limit=3)
    blocker = DiskBlocker(tmp_path / "index", pool, max_candidates=2)
    assert blocker.query(frame.iloc[:1]).shape == (0, 3)


def test_candidate_cap_and_feature_equality(tmp_path):
    pool, frame = store(tmp_path, [(f"S2-{i}", "acme", "12 main", "france", "acme", "12") for i in range(7)])
    build_index(pool, tmp_path / "index", bucket_limit=10)
    blocker = DiskBlocker(tmp_path / "index", pool, max_candidates=2)
    pairs = blocker.query(frame.iloc[:1])
    assert len(pairs) == 2
    assert len(set(pairs[:, 1])) == 2
    np.testing.assert_array_equal(pairs, blocker.query(frame.iloc[:1]))
    builder = BatchFeatures().fit(frame)
    result = builder.transform(frame.iloc[:1], pool, pairs)
    assert result.name_exact_normalized.eq(1).all()
    assert np.allclose(result.name_char3gram_tfidf_cosine, 1)
    assert np.isfinite(result.to_numpy()).all()
