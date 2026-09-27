"""Chunked normalization and memory-mapped Arrow record storage."""
from pathlib import Path
import json
import os
import time

import pandas as pd
import pyarrow as pa
import numpy as np

from ..normalization.normalizer import normalize_records

VERSION = 1
COLUMNS = ["entity_id", "name", "address", "country", "tokens", "numbers"]
SCHEMA = pa.schema([(c, pa.string()) for c in COLUMNS])


def signature(paths):
    return [{"name": Path(p).name, "bytes": Path(p).stat().st_size,
             "mtime_ns": Path(p).stat().st_mtime_ns} for p in paths]


def normalized_batches(path, chunksize=100_000):
    for raw in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize):
        df = normalize_records(raw)
        yield pd.DataFrame({
            "entity_id": df.entity_id,
            "name": df.business_name_normalized,
            "address": df.business_address_normalized,
            "country": df.country_normalized,
            "tokens": df.business_name_tokens.map(lambda x: " ".join(sorted(x))),
            "numbers": df.address_numbers.map(lambda x: " ".join(sorted(set(x)))),
        })


def prepare(path, destination):
    path, destination = Path(path), Path(destination)
    meta = destination.with_suffix(".json")
    expected = {"version": VERSION, "source": signature([path])}
    if destination.exists() and meta.exists() and json.loads(meta.read_text()) == expected:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".partial")
    total = 0
    start = time.perf_counter()
    with pa.OSFile(str(temp), "wb") as sink, pa.ipc.new_file(sink, SCHEMA) as writer:
        for frame in normalized_batches(path):
            writer.write_table(pa.Table.from_pandas(frame, schema=SCHEMA, preserve_index=False))
            total += len(frame)
            print(f"normalize {path.name}: {total:,} rows, {time.perf_counter()-start:.1f}s", flush=True)
    os.replace(temp, destination)
    meta.write_text(json.dumps(expected), encoding="utf-8")
    return destination


class RecordStore:
    def __init__(self, paths):
        self.maps = [pa.memory_map(str(p), "r") for p in paths]
        self.table = pa.concat_tables([pa.ipc.open_file(m).read_all() for m in self.maps])
        self.parts = self.table.to_batches()
        self.ends = np.cumsum([len(p) for p in self.parts])

    def __len__(self):
        return len(self.table)

    def take(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if not len(indices):
            return pd.DataFrame(columns=COLUMNS)
        order = np.argsort(indices, kind="stable")
        sorted_ids = indices[order]
        groups = np.searchsorted(self.ends, sorted_ids, side="right")
        chunks = []
        for group in np.unique(groups):
            base = int(self.ends[group-1]) if group else 0
            local = sorted_ids[groups == group] - base
            chunks.append(pa.Table.from_batches([self.parts[group]]).take(pa.array(local)))
        frame = pa.concat_tables(chunks).to_pandas()
        return frame.iloc[np.argsort(order)].reset_index(drop=True)

    def batches(self, size=100_000):
        for offset in range(0, len(self), size):
            yield offset, self.table.slice(offset, size).to_pandas()
