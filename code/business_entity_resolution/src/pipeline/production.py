"""Shared, bounded baseline training and full-test inference entry point."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import psutil

from ..blocking.disk_index import build_index, DiskBlocker
from ..features.batch_features import BatchFeatures
from ..models.baseline_model import MatchingModel
from ..shared.disk_store import prepare, RecordStore, signature
from ..shared.data_loader import ground_truth_to_dict
from ..calibration.calibrator import calibrate_threshold, apply_threshold
from ..evaluation.evaluator import full_evaluation_report

PAIR_COLUMNS = ["source1_entity_id", "candidate_entity_id", "match_probability"]


def memory():
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 1024**3


def save_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def prepared_paths(args, split):
    paths = [args.data_dir / split / f"{split}_source{i}.tsv" for i in (1, 2, 3)]
    fingerprint = signature(paths)
    # The directory is content-versioned by file metadata, preventing stale index reuse.
    from hashlib import sha256
    key = sha256(json.dumps(fingerprint).encode()).hexdigest()[:12]
    directory = args.work_dir / f"{split}-{key}"
    prepared = [prepare(p, directory / f"source{i}.arrow") for i, p in enumerate(paths, 1)]
    return prepared, directory, fingerprint


def stores(args, split):
    prepared, directory, fingerprint = prepared_paths(args, split)
    s1 = RecordStore(prepared[:1]); pool = RecordStore(prepared[1:])
    build_index(pool, directory / "index", args.bucket_limit)
    blocker = DiskBlocker(directory / "index", pool, args.max_candidates, args.max_keys)
    return s1, pool, blocker, fingerprint


def cohort_features(source, pool, blocker, feature_builder, batch_size):
    matrices = []; pairs_list = []
    for start in range(0, len(source), batch_size):
        chunk = source.iloc[start:start+batch_size].reset_index(drop=True)
        pairs = blocker.query(chunk)
        matrices.append(feature_builder.transform(chunk, pool, pairs))
        pairs[:, 0] += start
        pairs_list.append(pairs)
        print(f"features {min(start+batch_size,len(source)):,}/{len(source):,}; peak {memory():.2f} GiB", flush=True)
    return pd.concat(matrices, ignore_index=True), np.concatenate(pairs_list)


def score_frame(source, pool, pairs, probabilities):
    ids = pool.take(pairs[:, 1]).entity_id.to_numpy()
    return pd.DataFrame({PAIR_COLUMNS[0]: source.entity_id.to_numpy()[pairs[:, 0]],
                         PAIR_COLUMNS[1]: ids, PAIR_COLUMNS[2]: probabilities})


def report(source, scores, gt, pool_size, threshold):
    truth = {sid: gt[sid] for sid in source.entity_id}
    pairs = scores[PAIR_COLUMNS[:2]]
    result = full_evaluation_report(apply_threshold(scores, threshold), truth, pairs)
    result["reduction_ratio"] = 1 - len(pairs) / (len(source) * pool_size)
    return result


def train(args):
    started = time.perf_counter()
    s1, pool, blocker, fingerprint = stores(args, "train")
    # A fixed random S1 sample against the ENTIRE training candidate universe.
    # Sampling is explicit, saved in the artifact and never used at test inference.
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(len(s1), min(args.train_entities, len(s1)), replace=False)
    sampled = s1.take(selected)
    ids = set(sampled.entity_id)
    gt_parts = []
    for chunk in pd.read_csv(args.data_dir / "train/train_ground_truth.tsv", sep="\t",
                             dtype=str, keep_default_na=False, chunksize=100_000):
        gt_parts.append(chunk[chunk.source1_entity_id.isin(ids)])
    gt = ground_truth_to_dict(pd.concat(gt_parts))
    if set(gt) != ids:
        raise ValueError("Ground truth does not cover the full sampled Source 1 cohort")
    n = len(sampled)
    a, b, c = int(n*.7), int(n*.8), int(n*.9)
    cohorts = {"fit": sampled.iloc[:a].reset_index(drop=True),
               "early_stop": sampled.iloc[a:b].reset_index(drop=True),
               "calibration": sampled.iloc[b:c].reset_index(drop=True),
               "evaluation": sampled.iloc[c:].reset_index(drop=True)}
    if min(map(len, cohorts.values())) < 2:
        raise ValueError("Training requires at least 20 Source 1 entities")
    # Fit unsupervised vocabulary solely on fit cohort and its retrieved targets.
    train_pairs = blocker.query(cohorts["fit"])
    fit_records = pd.concat([cohorts["fit"], pool.take(np.unique(train_pairs[:, 1]))], ignore_index=True)
    builder = BatchFeatures().fit(fit_records)
    del fit_records, train_pairs
    datasets = {}
    for name, cohort in cohorts.items():
        print(f"processing {name}: {len(cohort):,} S1 records", flush=True)
        x, pairs = cohort_features(cohort, pool, blocker, builder, args.batch_size)
        targets = pool.take(pairs[:, 1]).entity_id.to_numpy()
        source_ids = cohort.entity_id.to_numpy()[pairs[:, 0]]
        truth_sets = {sid: set(gt[sid]) for sid in cohort.entity_id}
        labels = np.fromiter((cid in truth_sets[sid] for sid, cid in zip(source_ids, targets)), dtype=np.int8)
        datasets[name] = (x, pairs, labels)
    x, _, y = datasets["fit"]; ex, _, ey = datasets["early_stop"]
    if len(np.unique(y)) != 2 or not len(ex):
        raise ValueError("Insufficient candidate classes for training; increase --train-entities")
    config = {"model": {"type": "lightgbm", "num_leaves": 31, "learning_rate": .05,
                         "max_depth": -1, "min_child_samples": 20, "random_state": args.seed,
                         "n_estimators": args.rounds, "early_stopping_rounds": 20, "num_threads": args.threads}}
    model = MatchingModel(config)
    model.train(x, y, ex, ey)
    cx, cp, _ = datasets["calibration"]
    cs = score_frame(cohorts["calibration"], pool, cp, model.predict_proba(cx))
    calibration = calibrate_threshold(cs, {sid: gt[sid] for sid in cohorts["calibration"].entity_id},
                                      cohorts["calibration"].entity_id.tolist(),
                                      [float(t) for t in np.arange(.1, 1., .025)])
    threshold = calibration["best_threshold"]
    vx, vp, _ = datasets["evaluation"]
    vs = score_frame(cohorts["evaluation"], pool, vp, model.predict_proba(vx))
    metrics = report(cohorts["evaluation"], vs, gt, len(pool), threshold)
    metrics.update(threshold=threshold, calibration_f05=calibration["best_f05"],
                   sample_entities=n, full_source1_records=len(s1), candidate_pool_records=len(pool),
                   cohort_sizes={k:len(v) for k,v in cohorts.items()}, runtime_seconds=time.perf_counter()-started,
                   peak_memory_gib=memory(), training_pairs=len(x))
    args.model_dir.mkdir(parents=True, exist_ok=True)
    artifact = {"version": 1, "model": model, "features": builder, "threshold": threshold,
                "blocking": {k:getattr(args,k) for k in ("bucket_limit", "max_candidates", "max_keys")},
                "source_fingerprint": fingerprint, "seed": args.seed,
                "cohorts": {k:v.entity_id.tolist() for k,v in cohorts.items()}}
    temp = args.model_dir / "baseline.partial"
    with temp.open("wb") as f:
        pickle.dump(artifact, f, protocol=5)
    os.replace(temp, args.model_dir / "baseline.pkl")
    model.model.save_model(str(args.model_dir / "lightgbm.txt"))
    calibration["threshold_curve"].to_csv(args.model_dir / "threshold_curve.tsv", sep="\t", index=False)
    vs.to_csv(args.model_dir / "evaluation_scores.tsv", sep="\t", index=False)
    save_json(args.model_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


def infer(args):
    started = time.perf_counter()
    with (args.model_dir / "baseline.pkl").open("rb") as f:
        artifact = pickle.load(f)
    for key, value in artifact["blocking"].items():
        setattr(args, key, value)
    s1, pool, blocker, fingerprint = stores(args, "test")
    model, builder, threshold = artifact["model"], artifact["features"], artifact["threshold"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Chunk files are resumable; final names are only published after ALL rows finish.
    from hashlib import sha256
    model_hash = sha256((args.model_dir / "baseline.pkl").read_bytes()).hexdigest()
    run_key = sha256(json.dumps([fingerprint, model_hash, args.batch_size]).encode()).hexdigest()[:16]
    parts_dir = args.work_dir / ("inference-" + run_key)
    parts_dir.mkdir(parents=True, exist_ok=True)
    totals = Counter(); candidate_hist = Counter(); match_hist = Counter(); prob_hist = np.zeros(10, dtype=np.int64)
    peak = memory()
    for start, source in s1.batches(args.batch_size):
        state = parts_dir / f"{start:09d}.json"
        mpath = state.with_suffix(".matching"); cpath = state.with_suffix(".candidates")
        if state.exists() and mpath.exists() and cpath.exists():
            stats = json.loads(state.read_text())
        else:
            pairs = blocker.query(source)
            x = builder.transform(source, pool, pairs)
            probability = model.predict_proba(x) if len(x) else np.empty(0)
            if len(probability) != len(pairs) or not np.isfinite(probability).all():
                raise ValueError("Invalid model output")
            # IDs are fetched exclusively from the test S2/S3 record store.
            targets = pool.take(pairs[:, 1]).entity_id.to_numpy()
            counts = np.bincount(pairs[:, 0], minlength=len(source)) if len(pairs) else np.zeros(len(source), dtype=int)
            limits = np.r_[0, counts.cumsum()]
            local_matches = Counter()
            with mpath.open("w", encoding="utf-8", newline="") as mf, cpath.open("w", encoding="utf-8", newline="") as cf:
                for i, sid in enumerate(source.entity_id):
                    left, right = limits[i:i+2]
                    candidates = targets[left:right].tolist()
                    matches = targets[left:right][probability[left:right] >= threshold].tolist()
                    if len(set(candidates)) != len(candidates) or not all(cid.startswith(("S2-", "S3-")) for cid in candidates):
                        raise ValueError("Invalid or duplicate candidate IDs")
                    cf.write(sid + "\t" + ",".join(candidates) + "\n")
                    mf.write(sid + "\t" + ",".join(matches) + "\n")
                    local_matches[len(matches)] += 1
            stats = {"entities": len(source), "pairs": len(pairs), "matches": sum(k*v for k,v in local_matches.items()),
                     "candidate_hist": dict(Counter(map(int, counts))), "match_hist": dict(local_matches),
                     "prob_hist": np.histogram(probability, bins=np.linspace(0,1,11))[0].tolist(),
                     "peak_memory_gib": memory()}
            save_json(state, stats)
        totals.update({k:stats[k] for k in ("entities", "pairs", "matches")})
        candidate_hist.update({int(k):v for k,v in stats["candidate_hist"].items()})
        match_hist.update({int(k):v for k,v in stats["match_hist"].items()})
        prob_hist += stats["prob_hist"]
        peak = max(peak, stats["peak_memory_gib"])
        if start % (args.batch_size*10) == 0:
            print(f"inference {totals['entities']:,}/{len(s1):,}; pairs {totals['pairs']:,}; matches {totals['matches']:,}; {time.perf_counter()-started:.1f}s; peak {peak:.2f} GiB", flush=True)
    if totals["entities"] != len(s1):
        raise ValueError("Incomplete Source 1 coverage")
    for name, extension, header in (("matching_results.tsv", ".matching", "matched_entity_ids"),
                                     ("candidate_pairs.tsv", ".candidates", "candidate_entity_ids")):
        temp = args.output_dir / (name + ".partial")
        with temp.open("wb") as destination:
            destination.write(("source1_entity_id\t" + header + "\n").encode())
            import shutil
            for start in range(0, len(s1), args.batch_size):
                with (parts_dir / f"{start:09d}{extension}").open("rb") as part:
                    shutil.copyfileobj(part, destination)
        os.replace(temp, args.output_dir / name)
    counts = np.repeat(list(candidate_hist), list(candidate_hist.values()))
    metrics = dict(totals)
    metrics.update(zero_matches=match_hist[0], one_match=match_hist[1],
                   multiple_matches=sum(v for k,v in match_hist.items() if k>1),
                   avg_candidates=totals["pairs"]/len(s1), median_candidates=float(np.median(counts)),
                   p95_candidates=float(np.percentile(counts,95)), max_candidates=int(max(candidate_hist)),
                   candidate_reduction_ratio=1-totals["pairs"]/(len(s1)*len(pool)),
                   candidate_pool_records=len(pool), threshold=threshold, runtime_seconds=time.perf_counter()-started,
                   peak_memory_gib=peak, probability_histogram=prob_hist.tolist(),
                   match_count_histogram=dict(match_hist), source_fingerprint=fingerprint,
                   model_sha256=model_hash, official_validation="NOT YET RUN")
    save_json(args.output_dir / "inference_metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "infer", "prepare-test"])
    parser.add_argument("--data-dir", type=Path, default=Path("student_resource/dataset"))
    parser.add_argument("--work-dir", type=Path, default=Path("output/work"))
    parser.add_argument("--model-dir", type=Path, default=Path("output/model"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--train-entities", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--bucket-limit", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--max-keys", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.train_entities, args.batch_size, args.bucket_limit, args.max_candidates, args.max_keys) < 1:
        parser.error("Counts and limits must be positive")
    if args.mode == "prepare-test":
        prepared_paths(args, "test")
    else:
        (train if args.mode == "train" else infer)(args)
