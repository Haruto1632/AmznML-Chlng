"""Bounded disk-backed version of the existing token/phonetic/address index.

Oversized buckets are discarded before querying. Only a bounded union of the
rarest usable keys is materialized per S1. A deterministic evidence ranking
selects the FINAL pairs that will all receive model inference.
"""
from hashlib import blake2b
from pathlib import Path
import json
import time

import jellyfish
import numpy as np

DTYPE = np.dtype([("key", "<u8"), ("row", "<u4")])
VERSION = 1


def keys(record):
    tokens = sorted(record.tokens.split(), key=lambda t: (-len(t), t))[:6]
    numbers = sorted(record.numbers.split(), key=lambda t: (-len(t), t))[:2]
    result = [("t|" + t, 1) for t in tokens]
    if record.name:
        result.append(("e|" + " ".join(sorted(record.name.split())), 2))
    phonetic = next((t for t in tokens if t.isascii() and t.isalpha()), None)
    if phonetic:
        result.append(("p|" + record.country + "|" + jellyfish.soundex(phonetic), 4))
    for number in numbers:
        if len(number) >= 3:
            result.append(("a|" + number, 8))
        for token in tokens[:2]:
            result.append(("j|" + number + "|" + token, 8))
    return [(int.from_bytes(blake2b(k.encode(), digest_size=8).digest(), "little"), reason)
            for k, reason in dict(result).items()]


def build_index(store, directory, bucket_limit=128):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = directory / "index.json"
    expected = {"version": VERSION, "records": len(store), "bucket_limit": bucket_limit}
    if meta.exists() and json.loads(meta.read_text()) == expected:
        return
    raw_path = directory / "postings.tmp"
    start = time.perf_counter()
    with raw_path.open("wb") as output:
        for offset, frame in store.batches():
            postings = [(key, offset+i) for i, record in enumerate(frame.itertuples(index=False))
                        for key, _ in keys(record)]
            np.array(postings, dtype=DTYPE).tofile(output)
            print(f"index keys: {offset+len(frame):,}/{len(store):,}, {time.perf_counter()-start:.1f}s", flush=True)
    count = raw_path.stat().st_size // DTYPE.itemsize
    postings = np.memmap(raw_path, dtype=DTYPE, mode="r+", shape=(count,))
    print(f"sorting {count:,} disk-backed postings", flush=True)
    postings.sort(order="key", kind="quicksort")
    # One vector per unique bucket; no all-pairs matrix or Python bucket sets.
    starts = np.r_[0, np.flatnonzero(postings["key"][1:] != postings["key"][:-1]) + 1]
    lengths = np.diff(np.r_[starts, count])
    keep = lengths <= bucket_limit
    np.save(directory / "keys.npy", postings["key"][starts[keep]])
    offsets = np.r_[0, np.cumsum(lengths[keep], dtype=np.int64)]
    np.save(directory / "offsets.npy", offsets)
    # Stream compaction in blocks to avoid a 100M-element Python list.
    rows = np.lib.format.open_memmap(directory / "rows.npy", mode="w+", dtype="uint32", shape=(int(offsets[-1]),))
    position = 0
    for base in range(0, len(starts), 100_000):
        block_starts = starts[base:base+100_000]
        block_lengths = lengths[base:base+100_000]
        left = int(block_starts[0])
        right = int(block_starts[-1] + block_lengths[-1])
        mask = np.repeat(block_lengths <= bucket_limit, block_lengths)
        selected = postings["row"][left:right][mask]
        rows[position:position+len(selected)] = selected
        position += len(selected)
    rows.flush()
    del rows, postings
    raw_path.unlink()
    meta.write_text(json.dumps(expected), encoding="utf-8")
    print(f"index complete: {int(keep.sum()):,} buckets; {int(offsets[-1]):,} postings", flush=True)


class DiskBlocker:
    def __init__(self, directory, store, max_candidates=24, max_keys=4):
        self.keys = np.load(Path(directory) / "keys.npy", mmap_mode="r")
        self.offsets = np.load(Path(directory) / "offsets.npy", mmap_mode="r")
        self.rows = np.load(Path(directory) / "rows.npy", mmap_mode="r")
        self.store = store
        self.max_candidates = max_candidates
        self.max_keys = max_keys

    def query(self, source):
        """Return local S1 row, global S23 row, evidence mask. All bounds explicit."""
        records = list(source.itertuples(index=False))
        lists = [keys(r) for r in records]
        flat = [k for pairs in lists for k, _ in pairs]
        positions = np.searchsorted(self.keys, np.asarray(flat, dtype="uint64"))
        results = []
        cursor = 0
        for i, (record, record_keys) in enumerate(zip(records, lists)):
            usable = []
            for key, reason in record_keys:
                p = int(positions[cursor]); cursor += 1
                if p < len(self.keys) and self.keys[p] == key:
                    left, right = int(self.offsets[p]), int(self.offsets[p+1])
                    usable.append((right-left, left, right, reason))
            usable.sort()
            evidence = {}
            for size, left, right, reason in usable[:self.max_keys]:
                for rid in self.rows[left:right]:
                    rid = int(rid)
                    old_mask, old_weight = evidence.get(rid, (0, 0.))
                    evidence[rid] = (old_mask | reason, old_weight + 1 / size)
            if evidence:
                # Candidate retrieval is bounded at max_keys * bucket_limit.
                selected = sorted(evidence, key=lambda rid: (-evidence[rid][0].bit_count(), -evidence[rid][1], rid))[:self.max_candidates]
                results.extend((i, rid, evidence[rid][0]) for rid in selected)
        return np.asarray(results, dtype=np.int64).reshape(-1, 3)
