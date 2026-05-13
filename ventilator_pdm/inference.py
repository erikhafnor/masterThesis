"""Inference engine for deployed anomaly detection on Elisa 800 ventilators.

This module implements the operational scoring pipeline for predictive
maintenance at Helse Stavanger HF. It takes a trained anomaly detection
model (Isolation Forest or future autoencoder) and applies it to new
telemetry data to produce actionable alerts for clinical engineering staff.

Deployment Architecture
-----------------------

The inference engine sits between the feature pipeline and the alert
delivery layer:

    QuestDB (telemetry)
         |
         v
    extract.py  -->  data/fleet/fleet_YYYY-MM-DD.parquet
         |
         v
    features.py  -->  resample, pivot, derive, window
         |
         v
    inference.py  -->  InferenceEngine.score_latest()
         |                |
         |                v
         |          generate_alerts()  -->  alerts.csv / stdout
         |
         v
    feedback.py  -->  CMMS outcome labels (human-in-the-loop)
         |
         v
    retrain  -->  improved model (active learning cycle)

Scoring Methodology
-------------------
1. Raw telemetry (long format) is transformed through the standard feature
   pipeline: resample_then_pivot -> compute_derived_features ->
   filter_active_ventilation -> create_windows.
2. Each 30-minute window is summarized into statistical features (mean, std,
   min, max, delta) by the model's prepare_window_features().
3. The Isolation Forest scores each window; scores are negated so that
   higher values = more anomalous.
4. Fleet-percentile ranking contextualizes each device's score against the
   full fleet at the same time period.
5. Sustained exceedance of the 95th percentile (>=3 consecutive windows)
   triggers an alert, avoiding transient false alarms.

Alert Levels
------------
- WARNING: fleet percentile >= threshold for sustained_periods consecutive
  windows. Suggests scheduling inspection at next opportunity.
- CRITICAL: fleet percentile >= 0.99 OR anomaly score in top 1% of all
  historical scores. Suggests immediate inspection.

This module is designed to be run daily via cron or manual CLI invocation.
No complex infrastructure is required — just Python, the trained model file,
and access to the parquet data directory.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ventilator_pdm.evaluation import fleet_percentile_rank
from ventilator_pdm.extract import load_fleet_parquet
from ventilator_pdm.features import (
    compute_derived_features,
    create_windows,
    filter_active_ventilation,
    resample_then_pivot,
)
from ventilator_pdm.models.iforest import IForestModel, prepare_window_features
from ventilator_pdm.registry import FLEET_REGISTRY
from ventilator_pdm.variables import FEATURE_IDS, VAR_BY_ID

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Operational inference engine for anomaly detection.

    Loads a trained model (currently Isolation Forest) along with its
    scaler and feature column specification, then provides methods to
    score new telemetry data and generate actionable alerts.

    Parameters
    ----------
    model_path : Path
        Path to the trained model joblib file.
    feature_cols_path : Path or None
        Path to the numpy file containing feature column names.
        If None, looks for feature_cols.npy alongside the model file.

    Example
    -------
    >>> engine = InferenceEngine(Path("outputs/models/iforest/iforest.joblib"))
    >>> alerts = engine.score_latest(Path("data/fleet/"), lookback_days=1)
    >>> print(alerts[alerts["alert_level"] != "normal"])
    """

    def __init__(
        self,
        model_path: Path,
        feature_cols_path: Path | None = None,
    ) -> None:
        """Load a trained model and its associated feature column list.

        Args:
            model_path: Path to the trained ``IForestModel`` joblib file
                (e.g. ``outputs/models/iforest/iforest.joblib``).
            feature_cols_path: Path to the ``.npy`` file that contains the
                ordered list of feature column names used during training.
                If ``None``, the constructor looks for ``feature_cols.npy``
                in the same directory as ``model_path``.

        Raises:
            FileNotFoundError: If ``model_path`` or the resolved
                ``feature_cols_path`` does not exist.
        """
        self.model_path = Path(model_path)
        self.model = IForestModel.load(self.model_path)

        # Load feature column names
        if feature_cols_path is None:
            feature_cols_path = self.model_path.parent / "feature_cols.npy"
        self.feature_cols: list[str] = list(np.load(feature_cols_path, allow_pickle=True))

        logger.info(
            "InferenceEngine initialized: model=%s, %d feature columns",
            self.model_path.name,
            len(self.feature_cols),
        )

    def score_latest(
        self,
        data_dir: Path,
        lookback_days: int = 1,
    ) -> pd.DataFrame:
        """Score the most recent telemetry from parquet files.

        Selects fleet parquet files whose date suffix is within
        ``lookback_days + 1`` days of now, concatenates them, then delegates
        to :meth:`score_batch`.  When no files fall within the date window the
        method falls back to the ``lookback_days`` most-recently-modified
        files; if ``data_dir`` contains no parquet files at all a
        ``FileNotFoundError`` is raised.

        Args:
            data_dir: Directory containing files named
                ``fleet_YYYY-MM-DD.parquet``.
            lookback_days: Number of recent days to load and score.
                Defaults to 1 (yesterday + today).

        Returns:
            Scored windows DataFrame — see :meth:`score_batch` for the full
            column specification.

        Raises:
            FileNotFoundError: If ``data_dir`` contains no
                ``fleet_*.parquet`` files at all.
        """
        data_dir = Path(data_dir)
        cutoff = datetime.utcnow() - timedelta(days=lookback_days + 1)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        # Load only recent parquet files
        files = sorted(data_dir.glob("fleet_*.parquet"))
        recent_files = [f for f in files if f.stem.split("_", 1)[1] >= cutoff_str]

        if not recent_files:
            logger.warning(
                "No parquet files found after %s in %s, falling back to latest %d files",
                cutoff_str, data_dir, lookback_days,
            )
            recent_files = files[-lookback_days:] if files else []

        if not recent_files:
            raise FileNotFoundError(f"No fleet parquet files in {data_dir}")

        dfs = [pd.read_parquet(f) for f in recent_files]
        df_long = pd.concat(dfs, ignore_index=True)
        logger.info(
            "Loaded %d rows from %d recent files for scoring",
            len(df_long), len(recent_files),
        )

        return self.score_batch(df_long)

    def score_batch(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """Score arbitrary long-format telemetry data.

        Runs the full feature pipeline — ``resample_then_pivot``,
        ``compute_derived_features``, ``filter_active_ventilation``,
        ``create_windows`` — then scores each 30-minute window with the
        trained Isolation Forest model.  Scores are negated by
        :class:`~pdm.models.iforest.IForestModel` so that higher values
        indicate greater anomaly.  Fleet-percentile ranking is applied
        across all windows in the batch so that each score is contextualised
        relative to the rest of the fleet in the same time period.

        Args:
            df_long: Long-format telemetry DataFrame.  Required columns:
                ``timestamp``, ``device_serial``, ``variable_id``, ``value``.
                Optional ``bitfield_*`` columns are forwarded to the feature
                pipeline unchanged.

        Returns:
            DataFrame with one row per scored window containing:

            - ``device_serial`` (str): Telemetry serial of the ventilator.
            - ``window_start`` (Timestamp): Inclusive start of the window.
            - ``window_end`` (Timestamp): Exclusive end of the window.
            - ``anomaly_score`` (float): Negated Isolation Forest score;
                higher = more anomalous.
            - ``fleet_percentile`` (float): Rank of ``anomaly_score`` within
                the full batch (0.0 – 1.0).

            Returns an empty DataFrame with the same columns when
            ``df_long`` is empty.
        """
        if df_long.empty:
            logger.warning("Empty input DataFrame, returning empty scores")
            return pd.DataFrame(
                columns=[
                    "device_serial", "window_start", "window_end",
                    "anomaly_score", "fleet_percentile",
                ]
            )

        # Feature pipeline
        wide = resample_then_pivot(df_long, freq="1min")
        wide = compute_derived_features(wide)
        wide = filter_active_ventilation(wide)

        windows, metadata = create_windows(
            wide,
            window="30min",
            step="5min",
            feature_cols=self.feature_cols,
        )

        # Score
        scores = self.model.score(windows)

        scores_df = metadata[["device_serial", "window_start", "window_end"]].copy()
        scores_df["anomaly_score"] = scores
        scores_df["timestamp"] = scores_df["window_end"]

        # Fleet percentile ranking
        scores_df = fleet_percentile_rank(scores_df, score_col="anomaly_score")

        logger.info(
            "Scored %d windows across %d devices",
            len(scores_df),
            scores_df["device_serial"].nunique(),
        )
        return scores_df

    def generate_alerts(
        self,
        scores_df: pd.DataFrame,
        threshold_percentile: float = 0.95,
        sustained_periods: int = 3,
        critical_percentile: float = 0.99,
    ) -> pd.DataFrame:
        """Generate actionable alerts from scored windows.

        Groups windows by device and identifies *runs* of consecutive windows
        whose ``fleet_percentile`` exceeds ``threshold_percentile``.  A run
        must span at least ``sustained_periods`` consecutive windows to
        produce an alert; this sustained-exceedance criterion suppresses
        one-off transient spikes that do not indicate genuine degradation.

        Alert levels are assigned as follows:

        - ``"warning"``: run length >= ``sustained_periods`` and no window
          in the run exceeds ``critical_percentile``.
        - ``"critical"``: at least one window in the run has
          ``fleet_percentile >= critical_percentile``.
        - ``"normal"``: device has no qualifying run; a summary row is still
          included so callers receive a complete per-device status table.

        Args:
            scores_df: Output of :meth:`score_batch` or :meth:`score_latest`.
                Must contain ``device_serial``, ``window_start``,
                ``window_end``, ``anomaly_score``, and ``fleet_percentile``
                columns.
            threshold_percentile: Fleet percentile above which a window is
                considered anomalous for the purpose of WARNING detection.
                Defaults to 0.95.
            sustained_periods: Minimum number of consecutive above-threshold
                windows required to generate a WARNING or CRITICAL alert.
                At a 5-minute step this corresponds to 15 minutes of
                sustained anomaly.  Defaults to 3.
            critical_percentile: Fleet percentile above which a window
                contributes to CRITICAL escalation.  Defaults to 0.99.

        Returns:
            DataFrame with one row per device (one row per qualifying alert
            run plus one ``"normal"`` row for devices with no alert).
            Columns:

            - ``device_serial`` (str): Telemetry serial.
            - ``cmms_reg`` (int or None): CMMS registration number from
                :data:`~pdm.registry.FLEET_REGISTRY`.
            - ``alert_level`` (str): ``"critical"``, ``"warning"``, or
                ``"normal"``.
            - ``anomaly_score`` (float): Maximum anomaly score in the run
                (or overall device max for ``"normal"`` rows).
            - ``fleet_percentile`` (float): Maximum fleet percentile in the
                run (or overall device max for ``"normal"`` rows).
            - ``top_contributing_features`` (str): Comma-separated human-
                readable feature names from the model's feature column list.
            - ``recommended_action`` (str): Plain-language guidance for
                clinical engineering staff.
            - ``alert_start`` (Timestamp or None): Start of the first window
                in the alert run.
            - ``alert_end`` (Timestamp or None): End of the last window in
                the alert run.
            - ``n_anomalous_windows`` (int): Length of the alert run
                (0 for ``"normal"`` rows).

            Returns an empty DataFrame when ``scores_df`` is empty.
        """
        if scores_df.empty:
            return pd.DataFrame()

        alerts = []

        for serial, device_df in scores_df.groupby("device_serial"):
            device_df = device_df.sort_values("window_end")

            above_threshold = device_df["fleet_percentile"] >= threshold_percentile
            above_critical = device_df["fleet_percentile"] >= critical_percentile

            # Find sustained exceedances
            runs = above_threshold.astype(int).groupby(
                (above_threshold != above_threshold.shift()).cumsum()
            )

            has_alert = False
            for _, run in runs:
                if run.iloc[0] and len(run) >= sustained_periods:
                    run_data = device_df.loc[run.index]
                    max_score = run_data["anomaly_score"].max()
                    max_pctl = run_data["fleet_percentile"].max()
                    is_critical = above_critical.loc[run.index].any()

                    alert_level = "critical" if is_critical else "warning"

                    # Look up CMMS registration number
                    device_info = FLEET_REGISTRY.get(serial, {})
                    cmms_reg = device_info.get("cmms_reg", None)

                    # Top contributing features (by score magnitude in
                    # the most anomalous window)
                    top_features = self._identify_top_features(serial, run_data)

                    recommended = (
                        "Immediate inspection recommended — possible O2 sensor degradation"
                        if alert_level == "critical"
                        else "Schedule inspection at next opportunity — anomalous telemetry pattern detected"
                    )

                    alerts.append({
                        "device_serial": serial,
                        "cmms_reg": cmms_reg,
                        "alert_level": alert_level,
                        "anomaly_score": round(float(max_score), 4),
                        "fleet_percentile": round(float(max_pctl), 4),
                        "top_contributing_features": top_features,
                        "recommended_action": recommended,
                        "alert_start": run_data["window_start"].iloc[0],
                        "alert_end": run_data["window_end"].iloc[-1],
                        "n_anomalous_windows": len(run),
                    })
                    has_alert = True

            if not has_alert:
                # Include a "normal" row for completeness (optional)
                device_info = FLEET_REGISTRY.get(serial, {})
                cmms_reg = device_info.get("cmms_reg", None)
                alerts.append({
                    "device_serial": serial,
                    "cmms_reg": cmms_reg,
                    "alert_level": "normal",
                    "anomaly_score": round(float(device_df["anomaly_score"].max()), 4),
                    "fleet_percentile": round(float(device_df["fleet_percentile"].max()), 4),
                    "top_contributing_features": "",
                    "recommended_action": "No action required",
                    "alert_start": None,
                    "alert_end": None,
                    "n_anomalous_windows": 0,
                })

        result = pd.DataFrame(alerts)
        n_warnings = (result["alert_level"] == "warning").sum()
        n_critical = (result["alert_level"] == "critical").sum()
        logger.info(
            "Generated %d alerts: %d critical, %d warning, %d normal",
            len(result), n_critical, n_warnings,
            (result["alert_level"] == "normal").sum(),
        )
        return result

    def _identify_top_features(
        self,
        serial: str,
        run_data: pd.DataFrame,
        top_n: int = 3,
    ) -> str:
        """Identify likely top contributing features for an alert.

        Uses the feature column names to provide human-readable feature
        names. In the absence of SHAP/permutation importance (which
        would require the raw window data), we report the feature
        columns from the variable taxonomy for interpretability.

        Returns a comma-separated string of top feature names.
        """
        # Map feature column names to human-readable names
        feature_names = []
        for col in self.feature_cols[:top_n]:
            if col.startswith("var_"):
                vid = int(col.split("_")[1])
                var = VAR_BY_ID.get(vid)
                if var:
                    feature_names.append(f"{var.name} ({col})")
                else:
                    feature_names.append(col)
            elif col == "fio2_deviation":
                feature_names.append("FiO2 deviation (measured - setting)")
            else:
                feature_names.append(col)
        return ", ".join(feature_names)
