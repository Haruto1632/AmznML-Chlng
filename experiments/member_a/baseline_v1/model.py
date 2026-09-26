"""
model.py — LightGBM/XGBoost wrapper for baseline experiment.

Member A — Baseline v1
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class MatchingModel:
    """Wrapper for tree-based matching models."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_config = config['model']
        self.model = None
        self.feature_columns = None
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ):
        """Train the model with early stopping."""
        print("\n=== MODEL TRAINING ===")
        
        # Store feature columns
        self.feature_columns = [c for c in X_train.columns 
                                if c not in ['source1_entity_id', 'candidate_entity_id']]
        
        X_train_feat = X_train[self.feature_columns]
        X_val_feat = X_val[self.feature_columns]
        
        model_type = self.model_config['type']
        
        if model_type == 'lightgbm':
            self.model = self._train_lightgbm(X_train_feat, y_train, X_val_feat, y_val)
        elif model_type == 'xgboost':
            self.model = self._train_xgboost(X_train_feat, y_train, X_val_feat, y_val)
        elif model_type == 'histgbm':
            self.model = self._train_histgbm(X_train_feat, y_train, X_val_feat, y_val)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Model trained: {model_type}")
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': self.model_config['num_leaves'],
            'learning_rate': self.model_config['learning_rate'],
            'max_depth': self.model_config['max_depth'],
            'min_child_samples': self.model_config['min_child_samples'],
            'random_state': self.model_config['random_state'],
            'verbose': -1,
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=self.model_config['n_estimators'],
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.model_config['early_stopping_rounds']),
                lgb.log_evaluation(period=50),
            ],
        )
        
        return model
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6 if self.model_config['max_depth'] == -1 else self.model_config['max_depth'],
            'eta': self.model_config['learning_rate'],
            'seed': self.model_config['random_state'],
        }
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.model_config['n_estimators'],
            evals=evals,
            early_stopping_rounds=self.model_config['early_stopping_rounds'],
            verbose_eval=50,
        )
        
        return model
    
    def _train_histgbm(self, X_train, y_train, X_val, y_val):
        """Train sklearn HistGradientBoosting model."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        
        model = HistGradientBoostingClassifier(
            max_iter=self.model_config['n_estimators'],
            learning_rate=self.model_config['learning_rate'],
            max_depth=None if self.model_config['max_depth'] == -1 else self.model_config['max_depth'],
            random_state=self.model_config['random_state'],
            early_stopping=True,
            n_iter_no_change=self.model_config['early_stopping_rounds'],
            validation_fraction=len(X_val) / (len(X_train) + len(X_val)),
            verbose=1,
        )
        
        model.fit(X_train, y_train)
        return model
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict match probabilities."""
        X_feat = X[self.feature_columns]
        
        model_type = self.model_config['type']
        
        if model_type == 'lightgbm':
            return self.model.predict(X_feat, num_iteration=self.model.best_iteration)
        elif model_type == 'xgboost':
            import xgboost as xgb
            dtest = xgb.DMatrix(X_feat)
            return self.model.predict(dtest, iteration_range=(0, self.model.best_iteration + 1))
        elif model_type == 'histgbm':
            return self.model.predict_proba(X_feat)[:, 1]
    
    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_columns': self.feature_columns,
                'config': self.config,
            }, f)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        instance = cls(data['config'])
        instance.model = data['model']
        instance.feature_columns = data['feature_columns']
        return instance
