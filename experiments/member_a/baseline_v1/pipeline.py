"""
pipeline.py — End-to-end pipeline for baseline experiment.

Member A — Baseline v1

Import order matters: _project_paths must be first so that the shared
packages (shared/, evaluation/, calibration/, normalization/) are on
sys.path before anything tries to import them.
"""

from __future__ import annotations

# ── Path bootstrap (must be before any shared-package imports) ─────────────
import _project_paths  # noqa: F401  — side-effect: configures sys.path

# ── Stdlib ─────────────────────────────────────────────────────────────────
import sys
import time
from pathlib import Path

# ── Third-party ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Shared project modules (loaded as src.* so relative imports inside work) ─
from src.shared      import data_loader, schemas               # src/shared/
from src.evaluation  import evaluator                          # src/evaluation/
from src.calibration import calibrator                         # src/calibration/
from src.normalization.country_mapper import normalize_country # src/normalization/

# ── Experiment-local modules ───────────────────────────────────────────────
from normalization import clean_name, clean_address            # experiment dir
from blocking import MultiStrategyBlocker
from features import FeatureBuilder
from model import MatchingModel


class BaselinePipeline:
    """End-to-end entity resolution pipeline for baseline_v1."""

    def __init__(self, config: dict):
        self.config = config
        self.blocker = MultiStrategyBlocker(config)
        self.feature_builder = FeatureBuilder(config)
        self.model = MatchingModel(config)

        self.s1_train = None
        self.s2_train = None
        self.s3_train = None
        self.gt_train = None

        self.results = {}

    # ── Public entry point ─────────────────────────────────────────────────

    def run(self) -> dict:
        """Run the full experiment end-to-end. Returns the metrics dict."""
        start_time = time.time()

        print("=" * 70)
        print("BASELINE EXPERIMENT v1 — Member A")
        print("=" * 70)
        print(f"Train dir : {self.config['data']['train_dir']}")
        print(f"Output dir: {self.config['data']['output_dir']}")

        # 1. Load + normalise
        self._load_and_normalize()

        # 2. Generate candidates (blocking)
        candidate_pairs = self.blocker.block(
            self.s1_train, self.s2_train, self.s3_train
        )

        # 3. Blocking recall
        self._evaluate_blocking(candidate_pairs)

        # 4. Write candidate_pairs.tsv (challenge requirement)
        self._write_candidate_pairs(candidate_pairs)

        # 5. Features
        candidates_combined = pd.concat(
            [self.s2_train, self.s3_train], ignore_index=True
        )
        feature_df = self.feature_builder.build_features(
            candidate_pairs, self.s1_train, candidates_combined
        )

        # 6. Label + split
        X_train, X_val, y_train, y_val, val_s1_ids = self._prepare_training_data(
            feature_df
        )

        # 7. Train
        self.model.train(X_train, y_train, X_val, y_val)

        # 8. Predict on val set
        print("\n=== VALIDATION INFERENCE ===")
        # Recover the ID columns that were moved to the MultiIndex during split
        X_val_flat = X_val.reset_index()   # brings source1/candidate IDs back as columns
        proba = self.model.predict_proba(X_val_flat)
        predictions_df = pd.DataFrame({
            "source1_entity_id":   X_val_flat["source1_entity_id"],
            "candidate_entity_id": X_val_flat["candidate_entity_id"],
            "match_probability":   proba,
        })

        # 9. Calibrate threshold on val ground truth
        gt_dict = data_loader.ground_truth_to_dict(self.gt_train)
        gt_val   = {k: v for k, v in gt_dict.items() if k in val_s1_ids}

        cal_result = calibrator.calibrate_threshold(
            predictions_df,
            gt_val,
            list(val_s1_ids),
            self.config["calibration"]["thresholds"],
        )

        print(
            f"\nBest threshold: {cal_result['best_threshold']:.2f}"
            f"  (val F_0.5 = {cal_result['best_f05']:.4f})"
        )
        print("\nThreshold curve:")
        print(cal_result["threshold_curve"].to_string(index=False))

        # 10. Apply threshold → final predictions
        final_predictions = calibrator.apply_threshold(
            predictions_df, cal_result["best_threshold"]
        )
        for s1_id in val_s1_ids:          # guarantee every S1 has an entry
            final_predictions.setdefault(s1_id, [])

        # 11. Evaluate
        report = evaluator.full_evaluation_report(
            final_predictions, gt_val, candidate_pairs
        )
        elapsed = time.time() - start_time
        report["runtime_seconds"] = elapsed
        report["best_threshold"]  = cal_result["best_threshold"]
        self.results = report

        # 12. Write matching_results.tsv
        self._write_matching_results(final_predictions)

        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        evaluator.print_report(report)
        print(f"Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        return report

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_and_normalize(self) -> None:
        """Load TSVs and add normalised columns in-place."""
        print("\n=== DATA LOADING & NORMALIZATION ===")

        train_dir = self.config["data"]["train_dir"]
        self.s1_train, self.s2_train, self.s3_train, self.gt_train = (
            data_loader.load_all_train(train_dir)
        )

        print(f"  S1 train: {len(self.s1_train):>10,} records")
        print(f"  S2 train: {len(self.s2_train):>10,} records")
        print(f"  S3 train: {len(self.s3_train):>10,} records")

        expand = self.config["normalization"]["expand_legal_suffixes"]
        for label, df in [
            ("S1", self.s1_train),
            ("S2", self.s2_train),
            ("S3", self.s3_train),
        ]:
            print(f"  Normalising {label}…")
            self._normalize_dataframe(df, expand_suffixes=expand)

        print("Normalization complete.")

    def _normalize_dataframe(
        self, df: pd.DataFrame, *, expand_suffixes: bool = True
    ) -> None:
        """Add normalised columns to *df* in-place (vectorised where possible)."""
        # Pre-allocate columns
        df["name_normalized"]    = ""
        df["name_tokens"]        = None
        df["address_normalized"] = ""
        df["address_numbers"]    = None
        df["country_normalized"] = ""

        # Vectorise country (cheap)
        df["country_normalized"] = (
            df[schemas.COL_COUNTRY].fillna("").apply(normalize_country)
        )

        # Row-wise for name + address (require multiple return values)
        name_results = df[schemas.COL_BUSINESS_NAME].fillna("").apply(
            lambda n: clean_name(n, expand_suffixes=expand_suffixes)
        )
        df["name_normalized"] = name_results.apply(lambda t: t[0])
        df["name_tokens"]     = name_results.apply(lambda t: t[1])

        addr_results = df[schemas.COL_BUSINESS_ADDRESS].fillna("").apply(
            clean_address
        )
        df["address_normalized"] = addr_results.apply(lambda t: t[0])
        df["address_numbers"]    = addr_results.apply(lambda t: t[1])

    def _evaluate_blocking(self, candidate_pairs: pd.DataFrame) -> None:
        """Compute and store blocking recall."""
        print("\n=== BLOCKING RECALL ===")

        gt_dict = data_loader.ground_truth_to_dict(self.gt_train)

        # Build {s1_id: {cand_ids}} from candidate pairs
        cand_lookup: dict[str, set] = {}
        for row in candidate_pairs.itertuples(index=False):
            cand_lookup.setdefault(row.source1_entity_id, set()).add(
                row.candidate_entity_id
            )

        total_true = recalled = 0
        for s1_id, true_ids in gt_dict.items():
            cands = cand_lookup.get(s1_id, set())
            for tid in true_ids:
                total_true += 1
                if tid in cands:
                    recalled += 1

        blocking_recall = recalled / total_true if total_true else 0.0
        print(f"  True match pairs : {total_true:,}")
        print(f"  Recalled by blocking: {recalled:,}")
        print(f"  Blocking recall  : {blocking_recall:.4f}")

        n_cands = len(candidate_pairs)
        n_s1    = len(self.s1_train)
        n_s23   = len(self.s2_train) + len(self.s3_train)
        brute   = n_s1 * n_s23
        reduction = 1.0 - n_cands / brute if brute else 0.0
        avg_cands = n_cands / n_s1 if n_s1 else 0.0
        print(f"  Total candidates : {n_cands:,}")
        print(f"  Avg cands / S1   : {avg_cands:.1f}")
        print(f"  Reduction ratio  : {reduction:.6f}")

        self.results["blocking_recall"]   = blocking_recall
        self.results["avg_candidates"]    = avg_cands
        self.results["reduction_ratio"]   = reduction
        self.results["total_candidates"]  = n_cands

    def _prepare_training_data(
        self, feature_df: pd.DataFrame
    ) -> tuple:
        """Label pairs, do entity-level train/val split, return X/y splits."""
        print("\n=== PREPARING TRAINING DATA ===")

        gt_dict = data_loader.ground_truth_to_dict(self.gt_train)

        # Build a set of (s1_id, matched_id) for O(1) label lookup
        true_set: set[tuple] = set()
        for s1_id, matched_ids in gt_dict.items():
            for mid in matched_ids:
                true_set.add((s1_id, mid))

        feature_df = feature_df.copy()
        feature_df["label"] = [
            1 if (r.source1_entity_id, r.candidate_entity_id) in true_set else 0
            for r in feature_df.itertuples(index=False)
        ]

        n_pos = (feature_df["label"] == 1).sum()
        n_neg = (feature_df["label"] == 0).sum()
        print(f"  Positive pairs: {n_pos:,}")
        print(f"  Negative pairs: {n_neg:,}")
        print(f"  Positive rate : {n_pos / len(feature_df):.4f}")

        # Optional pair sampling (keeps memory manageable)
        sample_size = self.config["training"].get("sample_train_pairs")
        if sample_size and len(feature_df) > sample_size:
            print(f"  Sampling {sample_size:,} pairs…")
            feature_df = feature_df.sample(n=sample_size, random_state=42)

        # Entity-level split (prevents leakage across S1 entities)
        all_s1_ids = feature_df["source1_entity_id"].unique()
        train_ids, val_ids = train_test_split(
            all_s1_ids,
            test_size=self.config["training"]["val_split"],
            random_state=42,
        )
        val_ids_set = set(val_ids)

        train_mask = feature_df["source1_entity_id"].isin(train_ids)
        val_mask   = feature_df["source1_entity_id"].isin(val_ids)

        train_df = feature_df[train_mask].copy()
        val_df   = feature_df[val_mask].copy()

        id_cols      = ["source1_entity_id", "candidate_entity_id"]
        feature_cols = [c for c in feature_df.columns if c not in id_cols + ["label"]]

        # Keep ID columns in the index so predict_proba can recover them
        X_train = train_df[id_cols + feature_cols].set_index(id_cols)
        y_train = train_df["label"].values

        X_val   = val_df[id_cols + feature_cols].set_index(id_cols)
        y_val   = val_df["label"].values

        print(f"  Train pairs: {len(X_train):,}  ({(y_train == 1).sum():,} pos)")
        print(f"  Val pairs  : {len(X_val):,}  ({(y_val == 1).sum():,} pos)")

        return X_train, X_val, y_train, y_val, val_ids_set

    def _write_candidate_pairs(self, candidate_pairs: pd.DataFrame) -> None:
        """Write output/candidate_pairs.tsv."""
        out_dir = Path(self.config["data"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "candidate_pairs.tsv"

        # Group candidate IDs per S1 entity into comma-separated lists
        grouped = (
            candidate_pairs
            .groupby("source1_entity_id")["candidate_entity_id"]
            .apply(lambda ids: ",".join(dict.fromkeys(ids)))  # dedup, preserve order
            .reset_index()
            .rename(columns={"candidate_entity_id": "candidate_entity_ids"})
        )

        # Ensure every S1 train entity has a row (even singletons with no candidates)
        all_s1 = pd.DataFrame(
            {"source1_entity_id": self.s1_train[schemas.COL_ENTITY_ID]}
        )
        result = all_s1.merge(grouped, on="source1_entity_id", how="left")
        result["candidate_entity_ids"] = result["candidate_entity_ids"].fillna("")

        result.to_csv(out_path, sep="\t", index=False)
        print(f"\n  Wrote candidate_pairs.tsv  → {out_path}")

    def _write_matching_results(self, predictions: dict) -> None:
        """Write output/matching_results.tsv."""
        out_dir = Path(self.config["data"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "matching_results.tsv"

        rows = []
        for s1_id in self.s1_train[schemas.COL_ENTITY_ID]:
            matched = predictions.get(s1_id, [])
            deduped = list(dict.fromkeys(matched))
            rows.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(deduped),
            })

        pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
        print(f"  Wrote matching_results.tsv → {out_path}")
