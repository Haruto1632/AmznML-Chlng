"""
pipeline.py — End-to-end pipeline for baseline experiment.

Member A — Baseline v1
"""

import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path for imports
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from code.business_entity_resolution.src.shared import data_loader, schemas
from code.business_entity_resolution.src.evaluation import evaluator
from code.business_entity_resolution.src.calibration import calibrator
from code.business_entity_resolution.src.normalization.country_mapper import normalize_country

from normalization import clean_name, clean_address
from blocking import MultiStrategyBlocker
from features import FeatureBuilder
from model import MatchingModel


class BaselinePipeline:
    """End-to-end entity resolution pipeline."""
    
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
    
    def run(self):
        """Run the full experiment."""
        start_time = time.time()
        
        print("=" * 70)
        print("BASELINE EXPERIMENT v1 — Member A")
        print("=" * 70)
        
        # 1. Load and normalize data
        self._load_and_normalize()
        
        # 2. Generate candidates (blocking)
        candidate_pairs = self.blocker.block(self.s1_train, self.s2_train, self.s3_train)
        
        # 3. Evaluate blocking recall
        self._evaluate_blocking(candidate_pairs)
        
        # 4. Build features
        candidates_combined = pd.concat([self.s2_train, self.s3_train], ignore_index=True)
        feature_df = self.feature_builder.build_features(
            candidate_pairs, self.s1_train, candidates_combined
        )
        
        # 5. Prepare training data
        X_train, X_val, y_train, y_val = self._prepare_training_data(feature_df)
        
        # 6. Train model
        self.model.train(X_train, y_train, X_val, y_val)
        
        # 7. Predict on validation set
        print("\n=== VALIDATION INFERENCE ===")
        X_val_with_ids = X_val.copy()
        X_val_with_ids['source1_entity_id'] = X_val_with_ids.index.get_level_values(0)
        X_val_with_ids['candidate_entity_id'] = X_val_with_ids.index.get_level_values(1)
        X_val_with_ids = X_val_with_ids.reset_index(drop=True)
        
        y_pred_val = self.model.predict_proba(X_val_with_ids)
        predictions_df = pd.DataFrame({
            'source1_entity_id': X_val_with_ids['source1_entity_id'],
            'candidate_entity_id': X_val_with_ids['candidate_entity_id'],
            'match_probability': y_pred_val,
        })
        
        # 8. Calibrate threshold
        val_s1_ids = list(X_val_with_ids['source1_entity_id'].unique())
        gt_dict = data_loader.ground_truth_to_dict(self.gt_train)
        gt_val = {k: v for k, v in gt_dict.items() if k in val_s1_ids}
        
        cal_result = calibrator.calibrate_threshold(
            predictions_df,
            gt_val,
            val_s1_ids,
            self.config['calibration']['thresholds']
        )
        
        print(f"\nBest threshold: {cal_result['best_threshold']:.2f} (F_0.5 = {cal_result['best_f05']:.4f})")
        print("\nThreshold curve:")
        print(cal_result['threshold_curve'].to_string(index=False))
        
        # 9. Generate final predictions
        final_predictions = calibrator.apply_threshold(predictions_df, cal_result['best_threshold'])
        
        # Ensure all val S1 entities have predictions
        for s1_id in val_s1_ids:
            if s1_id not in final_predictions:
                final_predictions[s1_id] = []
        
        # 10. Evaluate
        report = evaluator.full_evaluation_report(final_predictions, gt_val, candidate_pairs)
        
        elapsed = time.time() - start_time
        report['runtime_seconds'] = elapsed
        report['best_threshold'] = cal_result['best_threshold']
        
        self.results = report
        
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        evaluator.print_report(report)
        print(f"Runtime: {elapsed:.1f}s")
        
        return report
    
    def _load_and_normalize(self):
        """Load data and apply normalization."""
        print("\n=== DATA LOADING & NORMALIZATION ===")
        
        train_dir = self.config['data']['train_dir']
        self.s1_train, self.s2_train, self.s3_train, self.gt_train = data_loader.load_all_train(train_dir)
        
        print(f"Train S1: {len(self.s1_train):,}")
        print(f"Train S2: {len(self.s2_train):,}")
        print(f"Train S3: {len(self.s3_train):,}")
        
        # Normalize all sources
        for df in [self.s1_train, self.s2_train, self.s3_train]:
            self._normalize_dataframe(df)
        
        print("Normalization complete.")
    
    def _normalize_dataframe(self, df: pd.DataFrame):
        """Add normalized columns to DataFrame in-place."""
        df['name_normalized'] = ''
        df['name_tokens'] = [set() for _ in range(len(df))]
        df['address_normalized'] = ''
        df['address_numbers'] = [[] for _ in range(len(df))]
        df['country_normalized'] = ''
        
        for idx, row in df.iterrows():
            # Name
            name_norm, name_tokens = clean_name(
                row[schemas.COL_BUSINESS_NAME],
                expand_suffixes=self.config['normalization']['expand_legal_suffixes']
            )
            df.at[idx, 'name_normalized'] = name_norm
            df.at[idx, 'name_tokens'] = name_tokens
            
            # Address
            addr_norm, addr_numbers = clean_address(row[schemas.COL_BUSINESS_ADDRESS])
            df.at[idx, 'address_normalized'] = addr_norm
            df.at[idx, 'address_numbers'] = addr_numbers
            
            # Country
            df.at[idx, 'country_normalized'] = normalize_country(row[schemas.COL_COUNTRY])
    
    def _evaluate_blocking(self, candidate_pairs: pd.DataFrame):
        """Compute blocking recall."""
        print("\n=== BLOCKING RECALL ===")
        
        gt_dict = data_loader.ground_truth_to_dict(self.gt_train)
        
        # Build candidate lookup
        cand_lookup = {}
        for _, row in candidate_pairs.iterrows():
            s1_id = row['source1_entity_id']
            cand_id = row['candidate_entity_id']
            cand_lookup.setdefault(s1_id, set()).add(cand_id)
        
        # Count true matches that survived blocking
        total_true_pairs = 0
        recalled_pairs = 0
        
        for s1_id, true_ids in gt_dict.items():
            cand_ids = cand_lookup.get(s1_id, set())
            for true_id in true_ids:
                total_true_pairs += 1
                if true_id in cand_ids:
                    recalled_pairs += 1
        
        blocking_recall = recalled_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
        
        print(f"True match pairs: {total_true_pairs:,}")
        print(f"Recalled pairs: {recalled_pairs:,}")
        print(f"Blocking recall: {blocking_recall:.4f}")
        
        self.results['blocking_recall'] = blocking_recall
    
    def _prepare_training_data(self, feature_df: pd.DataFrame):
        """Prepare labeled training data with train/val split."""
        print("\n=== PREPARING TRAINING DATA ===")
        
        gt_dict = data_loader.ground_truth_to_dict(self.gt_train)
        
        # Build positive/negative labels
        feature_df['label'] = 0
        
        for idx, row in feature_df.iterrows():
            s1_id = row['source1_entity_id']
            cand_id = row['candidate_entity_id']
            true_ids = gt_dict.get(s1_id, [])
            if cand_id in true_ids:
                feature_df.at[idx, 'label'] = 1
        
        n_pos = (feature_df['label'] == 1).sum()
        n_neg = (feature_df['label'] == 0).sum()
        
        print(f"Positive pairs: {n_pos:,}")
        print(f"Negative pairs: {n_neg:,}")
        print(f"Positive rate: {n_pos / len(feature_df):.4f}")
        
        # Sample if configured
        sample_size = self.config['training'].get('sample_train_pairs')
        if sample_size and len(feature_df) > sample_size:
            print(f"Sampling {sample_size:,} pairs...")
            feature_df = feature_df.sample(n=sample_size, random_state=42)
        
        # Entity-level train/val split
        all_s1_ids = feature_df['source1_entity_id'].unique()
        train_s1_ids, val_s1_ids = train_test_split(
            all_s1_ids,
            test_size=self.config['training']['val_split'],
            random_state=42,
        )
        
        train_mask = feature_df['source1_entity_id'].isin(train_s1_ids)
        val_mask = feature_df['source1_entity_id'].isin(val_s1_ids)
        
        train_df = feature_df[train_mask].copy()
        val_df = feature_df[val_mask].copy()
        
        # Extract X, y
        id_cols = ['source1_entity_id', 'candidate_entity_id']
        feature_cols = [c for c in feature_df.columns if c not in id_cols + ['label']]
        
        X_train = train_df[id_cols + feature_cols].set_index(id_cols)
        y_train = train_df['label']
        X_val = val_df[id_cols + feature_cols].set_index(id_cols)
        y_val = val_df['label']
        
        print(f"Train pairs: {len(X_train):,} ({(y_train == 1).sum():,} positive)")
        print(f"Val pairs: {len(X_val):,} ({(y_val == 1).sum():,} positive)")
        
        return X_train, X_val, y_train, y_val
