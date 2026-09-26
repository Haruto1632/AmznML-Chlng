"""
gbm_model.py — GBM matching model wrapper.

Owner: Member B  |  Branch: feature/matching-model

TODO (Member B): Implement GBMMatcherModel.
Prefer LightGBM, fall back to XGBoost, then HistGradientBoosting.
See TEAM_TASKS.md Task B-3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..shared.schemas import ALL_FEATURE_COLUMNS


class GBMMatcherModel:
    """
    Gradient Boosting Machine for pairwise entity matching.

    Outputs P(match | features) for each candidate pair.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.model = None
        self._backend = None

    def _get_backend(self):
        """Auto-detect available GBM backend."""
        backend = self.config.get("gbm_backend", "auto")
        if backend == "auto":
            try:
                import lightgbm  # noqa
                return "lightgbm"
            except ImportError:
                pass
            try:
                import xgboost  # noqa
                return "xgboost"
            except ImportError:
                pass
            return "histgbm"
        return backend

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None):
        """
        Train the GBM model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Should contain only feature columns (no ID columns).
        y : pd.Series
            Binary labels (1 = match, 0 = non-match).
        eval_set : tuple, optional
            (X_val, y_val) for early stopping.
        """
        # TODO: implement
        raise NotImplementedError("GBMMatcherModel.fit() not yet implemented. See Task B-3.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict match probability for each row.

        Returns
        -------
        np.ndarray of shape (n_samples,) with probabilities in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        # TODO: implement
        raise NotImplementedError("GBMMatcherModel.predict_proba() not yet implemented.")

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances as a DataFrame sorted by importance."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        # TODO: implement
        raise NotImplementedError

    def save(self, path: str):
        """Serialize model to disk."""
        import pickle, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "GBMMatcherModel":
        """Load a serialized model."""
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
