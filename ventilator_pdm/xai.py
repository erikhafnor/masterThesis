"""Explainable AI (XAI) module for ventilator anomaly detection.

This module provides SHAP-based and reconstruction-error-based feature
attribution for the two anomaly detection models in the Elisa 800 predictive
maintenance pipeline.

Explainer Framework:
    SHAP (SHapley Additive exPlanations) is the primary post-hoc explainer.
    TreeSHAP provides exact Shapley values for the Isolation Forest; the
    autoencoder uses intrinsic reconstruction-error decomposition instead.

Input Convention:
    All public functions accept scored alert windows (3-D numpy arrays of
    shape ``(n_windows, seq_len, n_features)``) together with the source
    feature column names (``feature_cols``).  Windows are produced by the
    ``pdm.features`` pipeline (resample, pivot, derive, window).

Output Convention:
    Per-feature attributions are returned as pandas DataFrames with columns
    ``feature_name``, ``human_name``, and a numeric attribution column
    (``mean_abs_shap`` or ``mean_reconstruction_error``), plus a ``rank``
    column (1 = most important).  Temporal (per-window) attributions are
    returned as numpy arrays of shape ``(n_windows, n_features)``.

Attribution Methods:
    1. **SHAP TreeExplainer for Isolation Forest** (Lundberg & Lee, 2017):
       TreeSHAP computes exact Shapley values in polynomial time for
       tree-based models.  For Isolation Forest each tree partitions the
       feature space using random splits; SHAP values quantify how each
       feature pushes a sample's anomaly score away from the expected
       (background) score.

       The IForest operates on 230 summary features (5 statistics x 46 raw
       telemetry features: mean, std, min, max, delta).  To recover
       attribution at the original 46-variable level, absolute SHAP values
       are summed across the 5 summary statistics per raw feature -- an
       approach analogous to the grouped SHAP method of Tang et al. (2024).

    2. **Reconstruction error decomposition for Autoencoder**:
       The CNN-LSTM autoencoder reconstructs normal ventilation patterns.
       Per-feature MSE between input and reconstruction provides intrinsic
       attribution without post-hoc methods.

Clinical Interpretability:
    For a biomedical engineer reviewing alerts, feature attributions answer
    "Why did the model flag this ventilator?"  For example, high attribution
    on ``var_635`` (fio2_measured) combined with ``var_2782`` (fio2_setting)
    indicates the model detected FiO2 set/measured divergence -- the hallmark
    of O2 sensor degradation.  High attribution on ``var_2098``
    (o2_supply_pressure) may point to gas supply issues rather than sensor
    fault, guiding the engineer to the correct maintenance action.

References:
    - Lundberg, S. M. & Lee, S.-I. (2017). A Unified Approach to
      Interpreting Model Predictions. NeurIPS.
    - Tang, Z. et al. (2024). Explainable predictive maintenance using SHAP.
    - Liu, F. T., Ting, K. M. & Zhou, Z.-H. (2008). Isolation Forest. ICDM.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ventilator_pdm.features import (
    compute_derived_features,
    create_windows,
    filter_active_ventilation,
    resample_then_pivot,
)
from ventilator_pdm.models.iforest import IForestModel, prepare_window_features
from ventilator_pdm.registry import KNOWN_FAILURES, REG_TO_SERIAL
from ventilator_pdm.variables import FEATURE_IDS, VAR_BY_ID

logger = logging.getLogger(__name__)

# Summary statistic names, matching the order in prepare_window_features
STAT_NAMES = ["mean", "std", "min", "max", "delta"]


def _feature_col_to_human(col: str) -> str:
    """Map a feature column name (e.g. 'var_635') to a human-readable name.

    Uses the VAR_BY_ID taxonomy from ventilator_pdm.variables.  Falls back to the
    raw column name for derived features (fio2_deviation, etc.) and
    bitfield columns.
    """
    if col.startswith("var_"):
        vid = int(col.split("_")[1])
        var = VAR_BY_ID.get(vid)
        if var:
            return var.name
    elif col == "fio2_deviation":
        return "FiO2 deviation"
    elif col == "o2_air_pressure_ratio":
        return "O2/Air pressure ratio"
    elif col == "volume_balance":
        return "Volume balance"
    elif col.startswith("bitfield_"):
        return col.replace("bitfield_", "").replace("_", " ")
    return col


def shap_feature_importance(
    model: IForestModel,
    windows: np.ndarray,
    feature_cols: list[str],
    n_background: int = 500,
) -> pd.DataFrame:
    """Compute SHAP-based feature importance for Isolation Forest windows.

    Uses TreeSHAP (Lundberg & Lee 2017) to compute exact Shapley values
    for the sklearn IsolationForest.  The 230 summary-feature SHAP values
    are aggregated back to the original 46 raw features by summing absolute
    SHAP values across the 5 summary statistics (mean, std, min, max, delta).

    Args:
        model: Trained IForest model.  ``model.model`` is the sklearn
            IsolationForest and ``model.scaler`` is the fitted
            StandardScaler.
        windows: 3-D array of shape ``(n_windows, seq_len, n_features)``.
        feature_cols: Names of the 46 raw feature columns.
        n_background: Number of background samples for the SHAP explainer.
            A representative background set is sampled from the input
            windows.

    Returns:
        DataFrame with columns ``[feature_name, human_name, mean_abs_shap,
        rank]``, sorted by rank (1 = most important).

    Raises:
        ImportError: If the ``shap`` package is not installed.
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP is required for tree-based feature attribution. "
            "Install with: pip install shap"
        )

    n_features = len(feature_cols)
    n_stats = len(STAT_NAMES)

    # Prepare 2D summary features and scale
    X = prepare_window_features(windows)
    col_median = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_median, np.where(nan_mask)[1])
    X_scaled = model.scaler.transform(X)

    # Background sample for explainer
    n_bg = min(n_background, X_scaled.shape[0])
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(X_scaled.shape[0], size=n_bg, replace=False)
    background = X_scaled[bg_idx]

    logger.info(
        "Computing SHAP values: %d windows, %d summary features, %d background",
        X_scaled.shape[0], X_scaled.shape[1], n_bg,
    )

    explainer = shap.TreeExplainer(model.model, data=background)
    shap_values = explainer.shap_values(X_scaled)

    # shap_values shape: (n_windows, 230)
    # Aggregate: sum |SHAP| across 5 stats per raw feature
    abs_shap = np.abs(shap_values)
    per_feature_shap = np.zeros((abs_shap.shape[0], n_features))

    for stat_idx in range(n_stats):
        start = stat_idx * n_features
        end = start + n_features
        per_feature_shap += abs_shap[:, start:end]

    mean_abs_shap = per_feature_shap.mean(axis=0)

    result = pd.DataFrame({
        "feature_name": feature_cols,
        "human_name": [_feature_col_to_human(c) for c in feature_cols],
        "mean_abs_shap": mean_abs_shap,
    })
    result = result.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    logger.info("Top SHAP features: %s", list(result.head(5)["human_name"]))
    return result


def shap_per_window(
    model: IForestModel,
    windows: np.ndarray,
    feature_cols: list[str],
    n_background: int = 500,
) -> np.ndarray:
    """Compute per-window, per-raw-feature SHAP attributions.

    Returns an array where each value is the sum of ``|SHAP|`` across the
    5 summary statistics for that window-feature pair.  Used for temporal
    attribution analysis (e.g. tracking how feature importance evolves in
    the days leading up to a failure).

    Args:
        model: Trained IForest model.
        windows: 3-D array of shape ``(n_windows, seq_len, n_features)``.
        feature_cols: Names of the 46 raw feature columns.
        n_background: Background sample size for TreeExplainer.

    Returns:
        Numpy array of shape ``(n_windows, n_features)``.

    Raises:
        ImportError: If the ``shap`` package is not installed.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP required. Install with: pip install shap")

    n_features = len(feature_cols)
    n_stats = len(STAT_NAMES)

    X = prepare_window_features(windows)
    col_median = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_median, np.where(nan_mask)[1])
    X_scaled = model.scaler.transform(X)

    n_bg = min(n_background, X_scaled.shape[0])
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(X_scaled.shape[0], size=n_bg, replace=False)
    background = X_scaled[bg_idx]

    explainer = shap.TreeExplainer(model.model, data=background)
    shap_values = explainer.shap_values(X_scaled)

    abs_shap = np.abs(shap_values)
    per_feature = np.zeros((abs_shap.shape[0], n_features))
    for stat_idx in range(n_stats):
        start = stat_idx * n_features
        end = start + n_features
        per_feature += abs_shap[:, start:end]

    return per_feature


def autoencoder_feature_attribution(
    model,  # AutoencoderModel
    windows: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compute per-feature reconstruction error from the autoencoder.

    The CNN-LSTM autoencoder reconstructs each input window; features
    with high mean squared error between input and reconstruction are
    those the model considers most anomalous.  This provides a natural,
    intrinsic attribution without requiring post-hoc explanation methods.

    Args:
        model (AutoencoderModel): Trained autoencoder exposing a
            ``score_per_feature()`` method that returns an array of shape
            ``(n_windows, n_features)``.
        windows: 3-D array of shape ``(n_windows, seq_len, n_features)``.
        feature_cols: Names of the feature columns.

    Returns:
        DataFrame with columns ``[feature_name, human_name,
        mean_reconstruction_error, rank]``, sorted by rank
        (1 = highest reconstruction error).
    """
    per_feature_mse = model.score_per_feature(windows)  # (n_windows, n_features)
    mean_mse = per_feature_mse.mean(axis=0)

    result = pd.DataFrame({
        "feature_name": feature_cols,
        "human_name": [_feature_col_to_human(c) for c in feature_cols],
        "mean_reconstruction_error": mean_mse,
    })
    result = result.sort_values("mean_reconstruction_error", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    logger.info("Top reconstruction-error features: %s", list(result.head(5)["human_name"]))
    return result


def _load_failure_data(
    data_dir: Path,
    serial: str,
    failure_date: date,
    pre_days: int,
    post_days: int,
) -> pd.DataFrame:
    """Load parquet data for a specific device around a failure date.

    Reads daily fleet parquet files that overlap the window
    [failure_date - pre_days, failure_date + post_days] and filters
    to the target device serial.

    Parameters
    ----------
    data_dir : Path
        Directory containing fleet_YYYY-MM-DD.parquet files.
    serial : str
        Telemetry device serial (15-char zero-padded).
    failure_date : date
        Known failure date.
    pre_days : int
        Days before failure to include.
    post_days : int
        Days after failure to include.

    Returns
    -------
    pd.DataFrame
        Long-format telemetry for the target device in the event window.
    """
    data_dir = Path(data_dir)
    start_date = failure_date - timedelta(days=pre_days + 1)
    end_date = failure_date + timedelta(days=post_days + 1)

    files = sorted(data_dir.glob("fleet_*.parquet"))
    relevant = []
    for f in files:
        try:
            file_date = date.fromisoformat(f.stem.split("_", 1)[1])
            if start_date <= file_date <= end_date:
                relevant.append(f)
        except (ValueError, IndexError):
            continue

    if not relevant:
        logger.warning("No parquet files found for %s around %s", serial, failure_date)
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in relevant]
    df = pd.concat(dfs, ignore_index=True)

    # Filter to target device
    df = df[df["device_serial"] == serial].copy()
    logger.info(
        "Loaded %d rows for %s around %s (%d files)",
        len(df), serial, failure_date, len(relevant),
    )
    return df


def failure_event_attribution(
    model_path: Path,
    data_dir: Path,
    method: str = "shap",
    pre_days: int = 14,
    post_days: int = 2,
) -> dict[int, pd.DataFrame]:
    """Compute feature attribution around each known failure event.

    For each of the known O2 sensor failures with telemetry data:

    1. Load parquet data in ``[failure - pre_days, failure + post_days]``.
    2. Run the feature pipeline (resample, pivot, derive, window).
    3. Score each window with the trained IForest.
    4. Compute per-window feature attribution (SHAP or reconstruction error).
    5. Annotate with days-to-failure for temporal analysis.

    Args:
        model_path: Path to the trained IForest model (``.joblib``).
        data_dir: Directory containing ``fleet_YYYY-MM-DD.parquet`` files.
        method: Attribution method -- ``"shap"`` for TreeSHAP or
            ``"reconstruction"`` for autoencoder per-feature MSE.
        pre_days: Days before the failure date to analyse.
        post_days: Days after the failure date to include.

    Returns:
        Dictionary keyed by ``cmms_reg`` (int).  Each value is a DataFrame
        with columns ``[window_start, feature_name, human_name,
        attribution_value, anomaly_score, days_to_failure]``.

    Raises:
        ValueError: If *method* is not a recognised attribution method.
    """
    model_path = Path(model_path)
    data_dir = Path(data_dir)

    # Load model and feature columns
    model = IForestModel.load(model_path)
    feature_cols_path = model_path.parent / "feature_cols.npy"
    feature_cols = list(np.load(feature_cols_path, allow_pickle=True))

    results = {}

    for failure in KNOWN_FAILURES:
        serial = failure["telemetry_serial"]
        cmms_reg = failure["cmms_reg"]
        fail_date = failure["date"]

        logger.info(
            "Processing failure: reg %d, serial %s, date %s",
            cmms_reg, serial, fail_date,
        )

        # Load device data around failure
        df = _load_failure_data(data_dir, serial, fail_date, pre_days, post_days)
        if df.empty:
            logger.warning("No data for reg %d — skipping", cmms_reg)
            continue

        # Feature pipeline
        try:
            wide = resample_then_pivot(df, freq="1min")
            wide = compute_derived_features(wide)
            wide = filter_active_ventilation(wide)

            # Ensure all expected feature columns exist (pad missing with NaN)
            for col in feature_cols:
                if col not in wide.columns:
                    wide[col] = np.nan
                    logger.debug("Padded missing column %s with NaN for reg %d", col, cmms_reg)

            windows, metadata = create_windows(
                wide, window="30min", step="5min", feature_cols=feature_cols,
            )
        except (ValueError, KeyError) as e:
            logger.warning("Feature pipeline failed for reg %d: %s", cmms_reg, e)
            continue

        if len(windows) < 2:
            logger.warning("Too few windows for reg %d (%d)", cmms_reg, len(windows))
            continue

        # Verify feature count matches model expectation
        n_expected = len(feature_cols)
        if windows.shape[2] != n_expected:
            logger.warning(
                "Reg %d: window has %d features, expected %d — skipping",
                cmms_reg, windows.shape[2], n_expected,
            )
            continue

        # Score windows
        scores = model.score(windows)

        # Compute per-window attribution
        if method == "shap":
            per_window_attr = shap_per_window(model, windows, feature_cols)
        else:
            raise ValueError(f"Method '{method}' not yet implemented for failure attribution")

        # Build per-window, per-feature long-format DataFrame
        fail_ts = pd.Timestamp(fail_date, tz="UTC")
        rows = []
        for w_idx in range(len(windows)):
            w_start = metadata.iloc[w_idx]["window_start"]
            if hasattr(w_start, "tz") and w_start.tz is None:
                w_start = w_start.tz_localize("UTC")
            days_to_fail = (fail_ts - w_start).total_seconds() / 86400

            for f_idx, col in enumerate(feature_cols):
                rows.append({
                    "window_start": w_start,
                    "feature_name": col,
                    "human_name": _feature_col_to_human(col),
                    "attribution_value": float(per_window_attr[w_idx, f_idx]),
                    "anomaly_score": float(scores[w_idx]),
                    "days_to_failure": days_to_fail,
                })

        event_df = pd.DataFrame(rows)
        results[cmms_reg] = event_df
        logger.info(
            "Reg %d: %d windows, %d attribution rows",
            cmms_reg, len(windows), len(event_df),
        )

    return results


def plot_failure_attribution(
    attribution_dict: dict[int, pd.DataFrame],
    output_dir: Path,
    top_n: int = 10,
) -> list[Path]:
    """Generate per-failure attribution plots and a global importance chart.

    For each failure event, creates a figure with two subplots:

    1. **Top:** Anomaly score trajectory with the failure date marked
       (vertical red dashed line).  Shows how the model's suspicion
       evolves over time.
    2. **Bottom:** Stacked area chart of the top-N feature attributions
       over time.  Reveals which telemetry signals drive the anomaly
       score and when their contribution begins to rise.

    A global horizontal bar chart of feature importance aggregated across
    all failure events is also saved.

    Args:
        attribution_dict: Output of :func:`failure_event_attribution`,
            keyed by ``cmms_reg``.
        output_dir: Directory to save the generated PNG files.
        top_n: Number of top features to include in the stacked area
            chart.

    Returns:
        List of ``Path`` objects pointing to all generated plot files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Collect global importance across all failures
    global_importance = {}

    for cmms_reg, df in attribution_dict.items():
        if df.empty:
            continue

        # Identify top features for this event (by mean attribution)
        feat_importance = (
            df.groupby("feature_name")["attribution_value"]
            .mean()
            .sort_values(ascending=False)
        )
        top_features = feat_importance.head(top_n).index.tolist()

        # Accumulate global importance
        for feat, val in feat_importance.items():
            global_importance[feat] = global_importance.get(feat, 0) + val

        # Per-window anomaly score (one value per window)
        score_timeline = (
            df.groupby("window_start")["anomaly_score"]
            .first()
            .sort_index()
        )

        # Per-window attribution for top features (pivot to wide)
        top_df = df[df["feature_name"].isin(top_features)].copy()
        attr_pivot = top_df.pivot_table(
            index="window_start",
            columns="human_name",
            values="attribution_value",
            aggfunc="first",
        ).sort_index()
        # Reorder columns by importance
        name_map = dict(zip(df["feature_name"], df["human_name"]))
        col_order = [name_map.get(f, f) for f in top_features if name_map.get(f, f) in attr_pivot.columns]
        attr_pivot = attr_pivot[col_order].fillna(0)

        # Failure timestamp
        fail_ts = pd.Timestamp(
            KNOWN_FAILURES[[f["cmms_reg"] for f in KNOWN_FAILURES].index(cmms_reg)]["date"],
            tz="UTC",
        )

        # V5 baseline (figure-style-guide.md, 2026-05-08): Courier monospace,
        # full frame, palette TEAL/FAILURE_RED. Scope via rc_context to avoid
        # leaking into callers' rcParams.
        # Font-floor (2026-05-13): all sizes ≥ 11 pt (figsize=6.3, tw=6.3 → scale=1).
        with plt.rc_context({
            "font.family": "monospace",
            "font.monospace": ["Courier New", "DejaVu Sans Mono"],
            "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
            "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
            "axes.spines.top": True, "axes.spines.right": True,
            "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
            "legend.frameon": False,
        }):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 5.0), sharex=True)

            # Top: anomaly score
            ax1.plot(score_timeline.index, score_timeline.values,
                     "-", color="#2a7f7f", linewidth=0.8, label="Anomaly score")
            ax1.axvline(fail_ts, color="#c0392b", linestyle="--", linewidth=1.0,
                        label="Failure date")
            ax1.set_ylabel("Anomaly score")
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.25)

            # Bottom: stacked area of top features
            ax2.stackplot(
                attr_pivot.index,
                [attr_pivot[c].values for c in attr_pivot.columns],
                labels=attr_pivot.columns,
                alpha=0.85,
            )
            ax2.axvline(fail_ts, color="#c0392b", linestyle="--", linewidth=1.0)
            ax2.set_ylabel("SHAP attribution (|SHAP| sum)")
            ax2.set_xlabel("Time")
            ax2.legend(loc="upper left", fontsize=11, ncol=2)
            ax2.grid(True, alpha=0.25)

            fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.10, hspace=0.10)
            path = output_dir / f"xai_reg_{cmms_reg}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        saved.append(path)
        logger.info("Saved attribution plot: %s", path)

    # Global importance bar chart
    if global_importance:
        gi_df = pd.DataFrame([
            {"feature_name": k, "human_name": _feature_col_to_human(k), "total_attribution": v}
            for k, v in global_importance.items()
        ]).sort_values("total_attribution", ascending=False).head(top_n * 2)

        with plt.rc_context({
            "font.family": "monospace",
            "font.monospace": ["Courier New", "DejaVu Sans Mono"],
            "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
            "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
            "axes.spines.top": True, "axes.spines.right": True,
            "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
            "legend.frameon": False,
        }):
            fig, ax = plt.subplots(figsize=(6.3, 4.0))
            ax.barh(
                gi_df["human_name"].values[::-1],
                gi_df["total_attribution"].values[::-1],
                color="#2a7f7f",
                edgecolor="#444444",
                linewidth=0.8,
                alpha=0.85,
            )
            ax.set_xlabel("Total |SHAP| attribution (summed across failures)")
            ax.grid(True, alpha=0.25, axis="x")
            fig.subplots_adjust(left=0.30, right=0.97, top=0.97, bottom=0.12)
            path = output_dir / "xai_global_importance.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        saved.append(path)
        logger.info("Saved global importance plot: %s", path)

    return saved


def generate_xai_report(
    model_path: Path,
    data_dir: Path,
    output_dir: Path,
    method: str = "shap",
    pre_days: int = 14,
    post_days: int = 2,
    top_n: int = 10,
) -> dict:
    """Run the full XAI analysis pipeline and save all outputs.

    Orchestrates failure-event attribution, plot generation, and summary
    table export.  Designed to be called from the CLI or programmatically
    from a notebook.

    Files saved to *output_dir*:
        - ``attribution_reg_{cmms_reg}.csv`` -- per-failure attribution data
        - ``xai_reg_{cmms_reg}.png`` -- per-failure attribution plot
        - ``xai_global_importance.png`` -- global feature importance bar chart
        - ``xai_summary.csv`` -- summary table of top features per failure
        - ``xai_summary.tex`` -- LaTeX version of the summary table

    Args:
        model_path: Path to the trained IForest model (``.joblib``).
        data_dir: Directory containing ``fleet_YYYY-MM-DD.parquet`` files.
        output_dir: Directory for all XAI outputs (created if missing).
        method: Attribution method (``"shap"`` or ``"reconstruction"``).
        pre_days: Days before the failure date to analyse.
        post_days: Days after the failure date to include.
        top_n: Number of top features to highlight in plots and the
            summary table.

    Returns:
        Dictionary with keys:

        - ``'attributions'`` -- ``dict[int, DataFrame]`` keyed by cmms_reg.
        - ``'plots'`` -- ``list[Path]`` of generated plot files.
        - ``'summary'`` -- ``DataFrame`` with per-failure top-N features.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== XAI Report Generation ===")
    logger.info("Model: %s", model_path)
    logger.info("Method: %s", method)
    logger.info("Window: [-%d days, +%d days]", pre_days, post_days)

    # Step 1: Compute attributions
    attributions = failure_event_attribution(
        model_path, data_dir, method=method,
        pre_days=pre_days, post_days=post_days,
    )

    if not attributions:
        logger.warning("No attributions computed — no failures with telemetry data?")
        return {"attributions": {}, "plots": [], "summary": pd.DataFrame()}

    # Step 2: Save per-failure CSVs
    for cmms_reg, df in attributions.items():
        csv_path = output_dir / f"attribution_reg_{cmms_reg}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved attribution CSV: %s", csv_path)

    # Step 3: Generate plots
    plots = plot_failure_attribution(attributions, output_dir, top_n=top_n)

    # Step 4: Build summary table
    summary_rows = []
    for cmms_reg, df in attributions.items():
        # Find failure info
        fail_info = [f for f in KNOWN_FAILURES if f["cmms_reg"] == cmms_reg][0]

        # Top features by mean attribution
        top = (
            df.groupby(["feature_name", "human_name"])["attribution_value"]
            .mean()
            .reset_index()
            .sort_values("attribution_value", ascending=False)
            .head(top_n)
        )

        n_windows = df["window_start"].nunique()
        mean_score = df.groupby("window_start")["anomaly_score"].first().mean()
        max_score = df.groupby("window_start")["anomaly_score"].first().max()

        for rank, (_, row) in enumerate(top.iterrows(), 1):
            summary_rows.append({
                "cmms_reg": cmms_reg,
                "failure_date": fail_info["date"],
                "description": fail_info["description"],
                "n_windows": n_windows,
                "mean_anomaly_score": round(float(mean_score), 4),
                "max_anomaly_score": round(float(max_score), 4),
                "feature_rank": rank,
                "feature_name": row["feature_name"],
                "human_name": row["human_name"],
                "mean_attribution": round(float(row["attribution_value"]), 6),
            })

    summary = pd.DataFrame(summary_rows)

    # Save summary CSV
    csv_path = output_dir / "xai_summary.csv"
    summary.to_csv(csv_path, index=False)
    logger.info("Saved summary CSV: %s", csv_path)

    # Save summary LaTeX
    try:
        # Compact version: top 5 features per failure
        latex_df = summary[summary["feature_rank"] <= 5][
            ["cmms_reg", "failure_date", "feature_rank", "human_name", "mean_attribution"]
        ].copy()
        latex_path = output_dir / "xai_summary.tex"
        latex_df.to_latex(latex_path, index=False, float_format="%.4f", caption=(
            "Top-5 SHAP feature attributions per detected failure event. "
            "Attribution values represent mean absolute SHAP contribution "
            "aggregated from 230 summary features to 46 raw telemetry variables."
        ), label="tab:xai_summary")
        logger.info("Saved summary LaTeX: %s", latex_path)
    except Exception as e:
        logger.warning("Could not save LaTeX table: %s", e)

    logger.info("=== XAI Report Complete: %d failures, %d plots ===", len(attributions), len(plots))
    return {"attributions": attributions, "plots": plots, "summary": summary}
