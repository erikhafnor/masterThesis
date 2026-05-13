"""Feature engineering for Elisa 800 ventilator telemetry.

Transforms long-format QuestDB telemetry into wide-format, windowed
feature arrays suitable for anomaly detection models (Autoencoder, IForest,
OCSVM).

Pipeline overview
-----------------
The canonical entry point is :func:`prepare_features`, which chains the
following steps:

1. :func:`resample_then_pivot` — resamples each variable to a common grid
   (default 1 min) per device and pivots from long to wide format.
2. :func:`compute_derived_features` — adds engineered signals such as
   ``fio2_deviation`` (measured minus set FiO2).
3. :func:`filter_active_ventilation` — drops rows where the standby
   variable indicates the device is not actively ventilating.
4. :func:`create_windows` — extracts overlapping sliding windows and
   returns a 3-D NumPy array.

Resampling semantics
--------------------
Variables in the Elisa 800 telemetry stream are sampled at different rates
(~22 s for some, ~58 s for others).  :func:`resample_then_pivot` resamples
each variable *independently* to a shared ``freq`` grid (default ``"1min"``)
using mean aggregation, then forward-fills gaps up to ``max_gap`` (default
``"5min"``).  This maximises co-presence of all variables before the final
join.

Input DataFrame shape (long format)
------------------------------------
Expected by most functions unless documented otherwise::

    timestamp      datetime64[ns] or datetime64[ns, UTC]
    device_serial  object   (15-digit serial string)
    variable_id    numeric  (int or Int64)
    value          float64  (numeric measurement)
    bitfield_*     Int8     (optional decoded boolean flags)

Output DataFrame shape (wide format)
--------------------------------------
Produced by pivot/resample functions::

    timestamp      datetime64[ns]
    device_serial  object
    var_<id>       float64   (one column per variable_id)
    bitfield_*     Int8      (one column per decoded flag, when present)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ventilator_pdm.variables import (
    BITFIELDS,
    FEATURE_IDS,
    FIO2_MEASURED_ID,
    FIO2_SETTING_ID,
    STANDBY_ID,
)

logger = logging.getLogger(__name__)


def resample_then_pivot(
    df: pd.DataFrame,
    freq: str = "1min",
    max_gap: str = "5min",
) -> pd.DataFrame:
    """Resample each variable to a common grid per device, then join into wide format.

    Unlike :func:`pivot_long_to_wide`, this function aligns multi-rate variables
    *before* joining, maximising the number of rows where all variables are
    present.  Variables sampled at ~22 s and ~58 s are resampled independently
    using mean aggregation, forward-filled within each series up to ``max_gap``,
    then merged on the shared ``freq`` index.

    Bitfield columns (``bitfield_*``), when present, are combined across source
    rows at each timestamp using ``max()`` so that a value of 1 from any source
    row wins over a default 0.

    Args:
        df: Long-format DataFrame with columns ``timestamp``,
            ``device_serial``, ``variable_id``, ``value``, and optionally
            one or more ``bitfield_*`` columns.
        freq: Target resampling frequency passed to
            ``pandas.DataFrame.resample``.  Defaults to ``"1min"``.
        max_gap: Maximum consecutive gap to forward-fill per variable,
            expressed as a pandas offset string (e.g. ``"5min"``).
            Gaps longer than this are left as ``NaN``.  Defaults to
            ``"5min"``.

    Returns:
        Wide-format DataFrame with columns ``timestamp``, ``device_serial``,
        ``var_<id>`` (one per variable_id), and ``bitfield_*`` (when present
        in the input).  One row per ``(device_serial, freq)`` bin.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["variable_id"] = df["variable_id"].astype(int)

    bitfield_cols = [c for c in df.columns if c.startswith("bitfield_")]
    max_periods = int(pd.Timedelta(max_gap) / pd.Timedelta(freq))

    result_parts = []

    for serial, device_df in df.groupby("device_serial"):
        var_frames: dict[str, pd.Series] = {}

        # Resample each numeric variable independently
        for vid, var_df in device_df.groupby("variable_id"):
            series = (
                var_df.set_index("timestamp")["value"]
                .sort_index()
                .resample(freq)
                .mean()
                .ffill(limit=max_periods)
            )
            var_frames[f"var_{int(vid)}"] = series

        if not var_frames:
            continue

        wide_device = pd.DataFrame(var_frames)
        wide_device["device_serial"] = serial

        # Merge bitfield columns: at each timestamp, multiple rows (from
        # var 801/802/803/804) may carry different decoded bits.  Use max()
        # to combine: 1 (OK) from any source row wins over 0 (NULL/default).
        if bitfield_cols:
            bf_df = (
                device_df[["timestamp"] + bitfield_cols]
                .set_index("timestamp")
                .sort_index()
                .groupby(level=0)[bitfield_cols]
                .max()
            )
            bf_resampled = bf_df.resample(freq).max().ffill(limit=max_periods)
            wide_device = wide_device.join(bf_resampled, how="left")

        result_parts.append(wide_device)

    result = pd.concat(result_parts).reset_index().rename(columns={"index": "timestamp"})

    logger.info(
        "Resample-then-pivot to wide format: %d rows, %d columns (freq=%s)",
        len(result), len(result.columns), freq,
    )
    return result


def pivot_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format telemetry to wide format with one column per variable_id.

    Uses ``pandas.pivot_table`` with ``aggfunc="mean"`` so that multiple
    readings for the same ``(timestamp, device_serial, variable_id)`` triplet
    are averaged.  Bitfield columns, when present, are merged by dropping
    duplicate ``(timestamp, device_serial)`` pairs (first occurrence kept after
    the merge).

    This function does *not* resample or align multi-rate variables; use
    :func:`resample_then_pivot` when variables are sampled at different rates.

    Args:
        df: Long-format DataFrame with required columns ``timestamp``,
            ``device_serial``, ``variable_id``, and ``value``, plus optional
            ``bitfield_*`` columns.

    Returns:
        Wide-format DataFrame with columns ``timestamp``, ``device_serial``,
        one ``var_<id>`` float64 column per distinct ``variable_id``, and
        ``bitfield_*`` columns (when present in the input).  Row index is
        reset to a contiguous integer range.
    """
    # Separate numeric values and bitfields
    bitfield_cols = [c for c in df.columns if c.startswith("bitfield_")]

    # Pivot numeric variable values
    numeric_df = df[["timestamp", "device_serial", "variable_id", "value"]].copy()
    numeric_df["variable_id"] = numeric_df["variable_id"].astype(int)

    wide = numeric_df.pivot_table(
        index=["timestamp", "device_serial"],
        columns="variable_id",
        values="value",
        aggfunc="mean",
    )
    wide.columns = [f"var_{int(c)}" for c in wide.columns]
    wide = wide.reset_index()

    # Merge bitfields if present
    if bitfield_cols:
        bitfield_df = (
            df[["timestamp", "device_serial"] + bitfield_cols]
            .drop_duplicates(subset=["timestamp", "device_serial"])
        )
        wide = wide.merge(bitfield_df, on=["timestamp", "device_serial"], how="left")

    logger.info(
        "Pivoted to wide format: %d rows, %d columns",
        len(wide), len(wide.columns),
    )
    return wide


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived engineering features to a wide-format telemetry DataFrame.

    Each derived feature is added only when all required source columns are
    present; missing source columns cause the feature to be silently skipped.

    Derived features added:

    - ``fio2_deviation``: measured FiO2 minus set FiO2
      (``var_635 - var_2782``). This is the primary signal for O2 sensor
      fault detection; a sustained negative deviation indicates the sensor
      is delivering less O2 than prescribed.
    - ``o2_air_pressure_ratio``: O2 supply pressure divided by air supply
      pressure (``var_2098 / var_2097``). Zero values in the denominator are
      replaced with ``NaN`` to avoid division-by-zero.
    - ``volume_balance``: inspiratory minus expiratory tidal volume
      (``var_2324 - var_2325``). Persistent imbalance may indicate a circuit
      leak.

    Args:
        df: Wide-format DataFrame as produced by :func:`pivot_long_to_wide`
            or :func:`resample_then_pivot`, with ``var_<id>`` columns for
            the relevant variable IDs.

    Returns:
        Copy of ``df`` with zero or more additional float64 columns appended
        (``fio2_deviation``, ``o2_air_pressure_ratio``, ``volume_balance``),
        depending on which source columns are present.
    """
    df = df.copy()

    fio2_meas_col = f"var_{FIO2_MEASURED_ID}"
    fio2_set_col = f"var_{FIO2_SETTING_ID}"

    if fio2_meas_col in df.columns and fio2_set_col in df.columns:
        df["fio2_deviation"] = df[fio2_meas_col] - df[fio2_set_col]
        logger.info("Computed fio2_deviation")

    if "var_2098" in df.columns and "var_2097" in df.columns:
        df["o2_air_pressure_ratio"] = df["var_2098"] / df["var_2097"].replace(0, np.nan)

    if "var_2324" in df.columns and "var_2325" in df.columns:
        df["volume_balance"] = df["var_2324"] - df["var_2325"]

    return df


def resample_and_fill(
    df: pd.DataFrame,
    freq: str = "5min",
    method: str = "ffill",
    max_gap: str = "30min",
) -> pd.DataFrame:
    """Resample wide-format data to a regular time grid per device.

    Resamples each device's time series independently using mean aggregation,
    then fills gaps according to ``method``.  Only forward-fill (``"ffill"``)
    is currently implemented; gaps longer than ``max_gap`` are left as
    ``NaN``.

    This function operates on *wide-format* input (one column per feature),
    unlike :func:`resample_then_pivot` which accepts long format.  Use this
    function after pivoting when a coarser resampling grid is needed for
    model input.

    Args:
        df: Wide-format DataFrame with columns ``timestamp``,
            ``device_serial``, and one or more feature columns.  The
            ``timestamp`` column is set as the index during processing and
            restored on return.
        freq: Target resampling frequency, e.g. ``"5min"``.  Passed
            directly to ``pandas.DataFrame.resample``.  Defaults to
            ``"5min"``.
        method: Gap-filling method.  Currently only ``"ffill"``
            (forward-fill) is supported.  Defaults to ``"ffill"``.
        max_gap: Maximum consecutive gap to forward-fill, expressed as a
            pandas offset string.  Gaps larger than this are left as
            ``NaN``.  Defaults to ``"30min"``.

    Returns:
        Wide-format DataFrame resampled to ``freq`` with ``timestamp``
        reset to a column and ``device_serial`` preserved.  The row index
        is reset to a contiguous integer range.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    result_parts = []
    for serial, group in df.groupby("device_serial"):
        group = group.drop(columns=["device_serial"])
        resampled = group.resample(freq).mean()

        if method == "ffill":
            max_gap_td = pd.Timedelta(max_gap)
            max_periods = int(max_gap_td / pd.Timedelta(freq))
            resampled = resampled.ffill(limit=max_periods)

        resampled["device_serial"] = serial
        result_parts.append(resampled)

    result = pd.concat(result_parts).reset_index()
    logger.info("Resampled to %s: %d rows", freq, len(result))
    return result


def filter_active_ventilation(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where the device is in standby mode.

    Drops all rows where the standby variable (``var_1889``) equals 1.
    Standby rows represent periods when the ventilator is powered but not
    actively treating a patient; including them would contaminate anomaly
    detection training with off-therapy patterns.

    If the standby column is absent from ``df`` (e.g. because ``STANDBY_ID``
    was not included in the extraction variable set), a warning is logged and
    the DataFrame is returned unchanged.

    Args:
        df: Wide-format DataFrame that may contain a ``var_1889`` column
            with values 0 (active) or 1 (standby).

    Returns:
        Copy of ``df`` with standby rows removed.  If the standby column is
        absent, returns ``df`` unmodified (same object, not a copy).
    """
    standby_col = f"var_{STANDBY_ID}"
    if standby_col not in df.columns:
        logger.warning("Standby column %s not found, skipping filter", standby_col)
        return df

    n_before = len(df)
    df = df[df[standby_col] != 1].copy()
    n_removed = n_before - len(df)
    logger.info("Removed %d standby rows (%.1f%%)", n_removed, 100 * n_removed / max(n_before, 1))
    return df


def create_windows(
    df: pd.DataFrame,
    window: str = "30min",
    step: str = "5min",
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create overlapping sliding windows over per-device time-series data.

    For each device, the time series is sliced into windows of length
    ``window`` with a stride of ``step``.  Window and step sizes are
    converted to integer row counts by dividing by the median inter-sample
    interval of the device's data, so the function works correctly even
    when ``df`` has already been resampled to a regular grid.

    Windows with fewer than 2 samples are discarded.  When all windows
    have the same length (the common case after regular-grid resampling),
    they are stacked directly via ``numpy.stack``; otherwise they are
    NaN-padded to the maximum observed length.

    Args:
        df: Wide-format DataFrame with columns ``timestamp``,
            ``device_serial``, and at least one feature column.  The
            ``timestamp`` column must be convertible by
            ``pandas.to_datetime``.
        window: Duration of each window, as a pandas offset string
            (e.g. ``"30min"``).  Defaults to ``"30min"``.
        step: Stride between successive window starts, as a pandas offset
            string.  Defaults to ``"5min"``.
        feature_cols: Ordered list of column names to include as features.
            Columns not present in ``df`` are silently dropped.  When
            ``None``, all columns whose names start with ``"var_"`` or
            ``"bitfield_"``, plus ``"fio2_deviation"``,
            ``"o2_air_pressure_ratio"``, and ``"volume_balance"``, are
            used automatically.

    Returns:
        A tuple ``(windows, metadata)`` where:

        - ``windows`` is a float32 ndarray of shape
            ``(n_windows, seq_len, n_features)``.
        - ``metadata`` is a DataFrame with one row per window and columns
            ``device_serial``, ``window_start``, ``window_end``, and
            ``n_samples``.

    Raises:
        ValueError: If no valid feature columns are found in ``df``.
        ValueError: If no windows can be created from the data (e.g.
            every device has fewer rows than one window length).
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.startswith("var_") or c.startswith("bitfield_")
                        or c in ("fio2_deviation", "o2_air_pressure_ratio", "volume_balance")]

    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("No feature columns found in DataFrame")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["device_serial", "timestamp"])

    window_td = pd.Timedelta(window)
    step_td = pd.Timedelta(step)

    all_windows = []
    metadata_rows: list[dict[str, Any]] = []

    for serial, group in df.groupby("device_serial"):
        group = group.set_index("timestamp")[feature_cols].sort_index()
        timestamps = group.index
        values = group.to_numpy(dtype=np.float32, na_value=np.nan)
        n_rows = len(group)

        if n_rows < 2:
            continue

        # Detect regular grid (allows fast integer-index windowing)
        median_dt = pd.Series(timestamps).diff().dropna().median()
        freq_td = pd.Timedelta(step) if median_dt <= step_td else median_dt

        # Compute window/step sizes in integer rows
        win_rows = max(1, int(round(window_td / freq_td)))
        step_rows = max(1, int(round(step_td / freq_td)))

        for start_idx in range(0, n_rows - win_rows + 1, step_rows):
            end_idx = start_idx + win_rows
            chunk = values[start_idx:end_idx]

            if len(chunk) >= 2:
                all_windows.append(chunk)
                metadata_rows.append({
                    "device_serial": serial,
                    "window_start": timestamps[start_idx],
                    "window_end": timestamps[min(end_idx, n_rows) - 1],
                    "n_samples": len(chunk),
                })

    if not all_windows:
        raise ValueError("No windows created — check data coverage")

    # Stack into uniform 3D array (all windows same length from integer indexing)
    max_len = max(w.shape[0] for w in all_windows)
    if all(w.shape[0] == max_len for w in all_windows):
        padded = np.stack(all_windows)
    else:
        padded = np.full((len(all_windows), max_len, len(feature_cols)), np.nan, dtype=np.float32)
        for i, w in enumerate(all_windows):
            padded[i, :w.shape[0], :] = w

    metadata = pd.DataFrame(metadata_rows)
    logger.info(
        "Created %d windows (%s, step %s) with %d features, max_len=%d",
        len(all_windows), window, step, len(feature_cols), max_len,
    )
    return padded, metadata


def prepare_features(
    df: pd.DataFrame,
    freq: str = "1min",
    window: str = "30min",
    step: str = "5min",
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Run the complete feature preparation pipeline on raw long-format telemetry.

    Chains four steps in order:

    1. :func:`resample_then_pivot` — resamples to ``freq`` and pivots to
       wide format.
    2. :func:`compute_derived_features` — adds ``fio2_deviation``,
       ``o2_air_pressure_ratio``, and ``volume_balance`` where source
       columns are available.
    3. :func:`filter_active_ventilation` — drops standby rows.
    4. :func:`create_windows` — extracts overlapping windows.

    Only Tier 1 and Tier 2 variable columns (IDs in ``FEATURE_IDS``) plus
    bitfield and derived columns are forwarded to :func:`create_windows`;
    context-only variables (Tier 3) are excluded.

    Args:
        df: Long-format raw telemetry DataFrame as returned by
            ``pdm.extract.load_fleet_parquet`` or equivalent, with columns
            ``timestamp``, ``device_serial``, ``variable_id``, ``value``,
            and optionally ``bitfield_*``.
        freq: Resampling frequency for :func:`resample_then_pivot`.
            Defaults to ``"1min"``.
        window: Window duration for :func:`create_windows`.
            Defaults to ``"30min"``.
        step: Window stride for :func:`create_windows`.
            Defaults to ``"5min"``.

    Returns:
        A tuple ``(windows, metadata, feature_cols)`` where:

        - ``windows`` is a float32 ndarray of shape
            ``(n_windows, seq_len, n_features)``.
        - ``metadata`` is a DataFrame with columns ``device_serial``,
            ``window_start``, ``window_end``, and ``n_samples``.
        - ``feature_cols`` is the ordered list of wide-format column names
            used as features (matches ``windows.shape[2]``).
    """
    wide = resample_then_pivot(df, freq=freq)
    wide = compute_derived_features(wide)
    wide = filter_active_ventilation(wide)

    feature_cols = [c for c in wide.columns if c.startswith("var_") or c.startswith("bitfield_")
                    or c in ("fio2_deviation", "o2_air_pressure_ratio", "volume_balance")]
    # Keep only Tier 1+2 var columns
    feature_cols = [
        c for c in feature_cols
        if not c.startswith("var_") or int(c.split("_")[1]) in FEATURE_IDS
    ]

    windows, metadata = create_windows(wide, window=window, step=step, feature_cols=feature_cols)
    return windows, metadata, feature_cols
