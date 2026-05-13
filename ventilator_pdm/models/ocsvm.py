"""Level 2b: One-Class SVM anomaly detection.

Same complexity tier as IForest — multivariate, unsupervised, operates on
the same n_features × 5 summary-statistic feature vectors from sliding windows.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDOneClassSVM
from sklearn.preprocessing import StandardScaler

from ventilator_pdm.models.iforest import prepare_window_features

logger = logging.getLogger(__name__)


class OCSVMModel:
    """One-Class SVM anomaly detector operating on windowed telemetry features.

    Uses `SGDOneClassSVM` (SGD-based linear classifier, no kernel) for
    scalability to large datasets. Scores are negated decision-function values
    so that higher values indicate greater anomaly likelihood, consistent with
    the project-wide convention.

    Attributes:
        model: The underlying fitted `SGDOneClassSVM` instance.
        scaler: The `StandardScaler` fitted on the training window features.
        max_subsamples: If not ``None``, training is performed on a random
            subsample of this many rows drawn from the full feature matrix.
    """

    def __init__(
        self,
        nu: float = 0.01,
        random_state: int = 42,
        max_subsamples: int | None = None,
    ):
        self.model = SGDOneClassSVM(
            nu=nu,
            random_state=random_state,
            tol=1e-4,
        )
        self.scaler = StandardScaler()
        self._fitted = False
        self.max_subsamples = max_subsamples

    def fit(self, windows: np.ndarray) -> "OCSVMModel":
        """Fit the OC-SVM on healthy training windows.

        Extracts summary statistics from the 3-D input, imputes NaN values
        with per-column medians, fits the internal scaler, then fits the model.
        Subsamples to `max_subsamples` rows before training if set.

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

        # Subsample for training if needed
        if self.max_subsamples is not None and X.shape[0] > self.max_subsamples:
            rng = np.random.RandomState(42)
            idx = rng.choice(X.shape[0], self.max_subsamples, replace=False)
            X_train = X[idx]
            logger.info(
                "OC-SVM training on %d/%d subsampled windows",
                len(X_train), X.shape[0],
            )
        else:
            X_train = X

        self.model.fit(X_train)
        self._fitted = True
        logger.info(
            "OC-SVM fitted on %d windows, %d features",
            X_train.shape[0], X_train.shape[1],
        )
        return self

    def fit_2d(self, X: np.ndarray) -> "OCSVMModel":
        """Fit the OC-SVM on a pre-computed 2-D feature matrix.

        Use this when summary statistics have already been extracted (e.g.,
        via `prepare_window_features`) so that the 3-D → 2-D flattening step
        can be skipped.

        Args:
            X: Feature matrix of shape (n_windows, n_features). NaN values
                are imputed with per-column medians.

        Returns:
            This instance (fitted in-place), enabling method chaining.
        """
        # Replace NaN with column median
        col_median = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X = X.copy()
            X[nan_mask] = np.take(col_median, np.where(nan_mask)[1])

        X = self.scaler.fit_transform(X)

        # Subsample for training if needed
        if self.max_subsamples is not None and X.shape[0] > self.max_subsamples:
            rng = np.random.RandomState(42)
            idx = rng.choice(X.shape[0], self.max_subsamples, replace=False)
            X_train = X[idx]
            logger.info(
                "OC-SVM training on %d/%d subsampled windows",
                len(X_train), X.shape[0],
            )
        else:
            X_train = X

        self.model.fit(X_train)
        self._fitted = True
        logger.info(
            "OC-SVM fitted on %d windows, %d features",
            X_train.shape[0], X_train.shape[1],
        )
        return self

    def score_2d(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for a pre-computed 2-D feature matrix.

        Args:
            X: Feature matrix of shape (n_windows, n_features). NaN values
                are imputed with per-column medians before scaling.

        Returns:
            1-D array of length n_windows. Higher values indicate greater
            anomaly likelihood (negated decision-function output).

        Raises:
            RuntimeError: If called before `fit` or `fit_2d`.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        # Replace NaN with column median
        col_median = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X = X.copy()
            X[nan_mask] = np.take(col_median, np.where(nan_mask)[1])
        X = self.scaler.transform(X)
        # decision_function: higher = more normal; negate for anomaly score
        raw = self.model.decision_function(X)
        return -raw

    def score(self, windows: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for the given windows.

        Args:
            windows: Windows to score, shape (n_windows, seq_len, n_features).

        Returns:
            1-D array of length n_windows. Higher values indicate greater
            anomaly likelihood (negated decision-function output).

        Raises:
            RuntimeError: If called before `fit` or `fit_2d`.
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
            RuntimeError: If called before `fit` or `fit_2d`.
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
        logger.info("Saved OC-SVM model to %s", path)

    @classmethod
    def load(cls, path: Path) -> "OCSVMModel":
        """Deserialize a previously saved model from disk.

        Args:
            path: Path to a file produced by `save`.

        Returns:
            A fully restored `OCSVMModel` instance ready for scoring.
        """
        data = joblib.load(path)
        obj = cls()
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj._fitted = True
        logger.info("Loaded OC-SVM model from %s", path)
        return obj
