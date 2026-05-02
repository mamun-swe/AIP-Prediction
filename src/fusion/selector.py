"""
Feature fusion and two-stage selection pipeline.

Stage 1 (filter):
    - Remove zero-variance features
    - Remove highly correlated features (Pearson |r| > threshold)

Stage 2 (embedded):
    - XGBoost feature importance ranking
    - Select top-k features

CRITICAL: Fit only on training data, transform both train and test.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold

from src.utils.logger import get_logger

log = get_logger(__name__)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Remove features whose Pearson |r| with another feature exceeds threshold."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.selected_indices_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None):
        log.info(f"Correlation filter: computing {X.shape[1]}×{X.shape[1]} matrix...")
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, 0.0)
        corr = np.abs(corr)

        to_drop = set()
        for i in range(corr.shape[0]):
            if i in to_drop:
                continue
            for j in range(i + 1, corr.shape[1]):
                if j in to_drop:
                    continue
                if corr[i, j] > self.threshold:
                    to_drop.add(j)  # drop the second, keep the first

        keep = [i for i in range(X.shape[1]) if i not in to_drop]
        self.selected_indices_ = np.array(keep)
        log.info(f"Correlation filter: {X.shape[1]} → {len(keep)} features "
                 f"(removed {len(to_drop)} correlated)")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class XGBImportanceSelector(BaseEstimator, TransformerMixin):
    """Select top-k features by XGBoost feature importance."""

    def __init__(self, k: int = 500, random_state: int = 42):
        self.k = k
        self.random_state = random_state
        self.selected_indices_: Optional[np.ndarray] = None
        self.importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        from xgboost import XGBClassifier
        log.info(f"XGB importance selector: fitting on {X.shape}...")
        scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False
        )
        clf.fit(X, y)
        self.importances_ = clf.feature_importances_
        k = min(self.k, X.shape[1])
        top_k = np.argsort(self.importances_)[::-1][:k]
        self.selected_indices_ = np.sort(top_k)
        log.info(f"XGB selector: selected top {k} of {X.shape[1]} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)


class FeatureFusionPipeline:
    """
    Full feature fusion and selection pipeline.

    Usage:
        pipeline = FeatureFusionPipeline(n_features=500)
        X_train_final = pipeline.fit_transform(X_hc_train, X_plm_train, y_train)
        X_test_final  = pipeline.transform(X_hc_test, X_plm_test)
        pipeline.save("results/features/fused/pipeline.pkl")
    """

    def __init__(self, variance_threshold: float = 0.01,
                 correlation_threshold: float = 0.95,
                 n_features: int = 500,
                 random_state: int = 42):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.n_features = n_features
        self.random_state = random_state

        self._var_filter = VarianceThreshold(threshold=variance_threshold)
        self._corr_filter = CorrelationFilter(threshold=correlation_threshold)
        self._importance_selector = XGBImportanceSelector(
            k=n_features, random_state=random_state
        )
        self._fitted = False
        self.feature_names_after_selection_: Optional[List[str]] = None

    def _fuse(self, X_hc: np.ndarray,
              X_plm: np.ndarray) -> np.ndarray:
        """Concatenate hand-crafted and PLM features."""
        assert X_hc.shape[0] == X_plm.shape[0], \
            f"Sample count mismatch: {X_hc.shape[0]} vs {X_plm.shape[0]}"
        return np.concatenate([X_hc, X_plm], axis=1)

    def fit_transform(self, X_hc: np.ndarray, X_plm: np.ndarray,
                      y: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit on training data and return transformed training features."""
        log.info(f"Fusing features: HC={X_hc.shape}, PLM={X_plm.shape}")
        X = self._fuse(X_hc, X_plm)
        log.info(f"  Fused: {X.shape}")

        # Stage 1a: variance filter
        X = self._var_filter.fit_transform(X)
        log.info(f"  After variance filter: {X.shape[1]} features")

        # Stage 1b: correlation filter
        X = self._corr_filter.fit_transform(X)
        log.info(f"  After correlation filter: {X.shape[1]} features")

        # Stage 2: importance-based selection
        X = self._importance_selector.fit_transform(X, y)
        log.info(f"  After XGB selection: {X.shape[1]} features")

        self._fitted = True

        # Track feature names if provided
        if feature_names is not None:
            try:
                names = np.array(feature_names)
                after_var = names[self._var_filter.get_support()]
                after_corr = after_var[self._corr_filter.selected_indices_]
                self.feature_names_after_selection_ = list(
                    after_corr[self._importance_selector.selected_indices_]
                )
            except Exception as e:
                log.warning(f"Could not track feature names: {e}")

        return X

    def transform(self, X_hc: np.ndarray,
                  X_plm: np.ndarray) -> np.ndarray:
        """Transform new data using fitted pipeline (test set)."""
        assert self._fitted, "Call fit_transform on training data first."
        X = self._fuse(X_hc, X_plm)
        X = self._var_filter.transform(X)
        X = self._corr_filter.transform(X)
        X = self._importance_selector.transform(X)
        return X

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info(f"Saved feature pipeline → {path}")

    @staticmethod
    def load(path: str) -> "FeatureFusionPipeline":
        pipeline = joblib.load(path)
        log.info(f"Loaded feature pipeline from {path}")
        return pipeline

    def save_features(self, X: np.ndarray, y: np.ndarray,
                      path: str, split_name: str = "train"):
        """Save final feature matrix and labels as npz."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, X=X, y=y, split=split_name)
        log.info(f"Saved {split_name} features → {path} (shape={X.shape})")

    @staticmethod
    def load_features(path: str) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(path)
        return data["X"], data["y"]
