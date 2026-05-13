"""Level 2: Isolation Forest anomaly detection.

Trains on windowed Tier 1+2 features, fleet-only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def prepare_window_features(windows: np.ndarray) -> np.ndarray:
    """Flatten 3D windows into a 2D summary-statistic feature matrix.

    For each window, computes mean, std, min, max, and last-minus-first delta
    per feature channel, then concatenates them into a single row. NaN values
    are propagated through the nanX functions and should be imputed by the
    caller before use.

    Args:
        windows: Array of shape (n_windows, seq_len, n_features) containing
            time-series windows. Values may include NaN.

    Returns:
        Float32 array of shape (n_windows, n_features * 5) containing the
        concatenated per-feature summary statistics.
    """
    n_windows, seq_len, n_features = windows.shape
    stats = []

    for i in range(n_windows):
        w = windows[i]
        with np.errstate(all="ignore"):
            feat_mean = np.nanmean(w, axis=0)
            feat_std = np.nanstd(w, axis=0)
            feat_min = np.nanmin(w, axis=0)
            feat_max = np.nanmax(w, axis=0)
            feat_delta = w[-1] - w[0]
            feat_delta = np.where(np.isnan(feat_delta), 0, feat_delta)

        row = np.concatenate([feat_mean, feat_std, feat_min, feat_max, feat_delta])
        stats.append(row)

    return np.array(stats, dtype=np.float32)


class IForestModel:
    """Isolation Forest anomaly detector operating on windowed telemetry features.

    Wraps scikit-learn's `IsolationForest` with an integrated `StandardScaler`
    and handles the 3-D-to-2-D flattening via `prepare_window_features`. Scores
    are negated decision-function values so that higher scores indicate greater
    anomaly likelihood, consistent with the project-wide convention.

    Attributes:
        model: The underlying fitted `IsolationForest` instance.
        scaler: The `StandardScaler` fitted on the training window features.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.01,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, windows: np.ndarray) -> "IForestModel":
        """Fit the Isolation Forest on healthy training windows.

        Extracts summary statistics from the 3-D input, imputes NaN values
        with per-column medians, fits the internal scaler, then fits the model.

        Args:
            windows: Healthy training windows, shape (n_windows, seq_len,
                n_features).

        Returns:
            This instance (fitted in-place), enabling method chaining.
        """
        X = prepare_window_features(windows)
        # Replace NaN with column median
        col_median = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_median, np.where(nan_mask)[1])

        X = self.scaler.fit_transform(X)
        self.model.fit(X)
        self._fitted = True
        logger.info("IForest fitted on %d windows, %d features", X.shape[0], X.shape[1])
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for the given windows.

        Args:
            windows: Windows to score, shape (n_windows, seq_len, n_features).

        Returns:
            1-D array of length n_windows. Higher values indicate greater
            anomaly likelihood (negated decision-function output).

        Raises:
            RuntimeError: If called before `fit`.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        X = prepare_window_features(windows)
        col_median = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_median, np.where(nan_mask)[1])
        X = self.scaler.transform(X)
        # decision_function: higher = more normal; negate for anomaly score
        raw = self.model.decision_function(X)
        return -raw

    def predict(self, windows: np.ndarray) -> np.ndarray:
        """Predict binary anomaly labels for the given windows.

        Args:
            windows: Windows to classify, shape (n_windows, seq_len, n_features).

        Returns:
            1-D integer array of length n_windows: -1 for anomaly, 1 for normal.

        Raises:
            RuntimeError: If called before `fit`.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        X = prepare_window_features(windows)
        col_median = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_median, np.where(nan_mask)[1])
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        """Serialize the fitted model and scaler to disk.

        Args:
            path: Destination file path. Parent directories are created if
                they do not exist. Serialization uses `joblib.dump`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        logger.info("Saved IForest model to %s", path)

    @classmethod
    def load(cls, path: Path) -> "IForestModel":
        """Deserialize a previously saved model from disk.

        Args:
            path: Path to a file produced by `save`.

        Returns:
            A fully restored `IForestModel` instance ready for scoring.
        """
        data = joblib.load(path)
        obj = cls()
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj._fitted = True
        logger.info("Loaded IForest model from %s", path)
        return obj
