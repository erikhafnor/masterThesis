"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from ventilator_pdm.features import (
    compute_derived_features,
    create_windows,
    filter_active_ventilation,
    pivot_long_to_wide,
    resample_and_fill,
    resample_then_pivot,
)


def _make_long_df(n_devices=2, n_timestamps=100, variable_ids=(635, 2782)):
    """Create synthetic long-format telemetry data."""
    rows = []
    for d in range(n_devices):
        serial = f"00000000{d:07d}"
        for t in range(n_timestamps):
            ts = pd.Timestamp("2025-12-01") + pd.Timedelta(minutes=5 * t)
            for vid in variable_ids:
                rows.append({
                    "timestamp": ts,
                    "device_serial": serial,
                    "variable_id": vid,
                    "value": 0.4 + np.random.normal(0, 0.01),
                })
    return pd.DataFrame(rows)


def test_pivot_long_to_wide():
    df = _make_long_df(n_devices=1, n_timestamps=10)
    wide = pivot_long_to_wide(df)

    assert "var_635" in wide.columns
    assert "var_2782" in wide.columns
    assert "device_serial" in wide.columns
    assert len(wide) == 10


def test_compute_derived_features():
    df = _make_long_df(n_devices=1, n_timestamps=10)
    wide = pivot_long_to_wide(df)
    derived = compute_derived_features(wide)

    assert "fio2_deviation" in derived.columns
    # Deviation should be small (both vars have similar values)
    assert derived["fio2_deviation"].abs().max() < 0.1


def test_fio2_deviation_signal():
    """Test that FiO2 deviation correctly captures the difference."""
    df = _make_long_df(n_devices=1, n_timestamps=10, variable_ids=(635, 2782))
    wide = pivot_long_to_wide(df)
    # Set a known deviation
    wide["var_635"] = 0.36
    wide["var_2782"] = 0.40
    derived = compute_derived_features(wide)

    np.testing.assert_allclose(derived["fio2_deviation"].values, -0.04, atol=1e-10)


def test_filter_active_ventilation():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-12-01", periods=10, freq="5min"),
        "device_serial": "000000000000001",
        "var_1889": [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
        "var_635": range(10),
    })
    filtered = filter_active_ventilation(df)
    assert len(filtered) == 7  # 3 standby rows removed


def test_resample_and_fill():
    timestamps = pd.date_range("2025-12-01", periods=20, freq="3min")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "device_serial": "000000000000001",
        "var_635": np.random.randn(20),
    })
    resampled = resample_and_fill(df, freq="5min")
    assert len(resampled) > 0
    # Check regular spacing
    diffs = resampled["timestamp"].diff().dropna()
    assert (diffs == pd.Timedelta("5min")).all()


def _make_multirate_long_df(n_devices=1, duration_minutes=60):
    """Create synthetic long-format data with realistic multi-rate sampling.

    var_635 (FiO2 measured) at ~58s intervals,
    var_2782 (FiO2 setting) at ~22s intervals.
    """
    rows = []
    for d in range(n_devices):
        serial = f"00000000{d:07d}"
        start = pd.Timestamp("2025-12-01")
        # var_635 at 58s
        t = start
        end = start + pd.Timedelta(minutes=duration_minutes)
        while t < end:
            rows.append({"timestamp": t, "device_serial": serial,
                         "variable_id": 635, "value": 35.0 + np.random.normal(0, 0.5)})
            t += pd.Timedelta(seconds=58)
        # var_2782 at 22s
        t = start
        while t < end:
            rows.append({"timestamp": t, "device_serial": serial,
                         "variable_id": 2782, "value": 35.0})
            t += pd.Timedelta(seconds=22)
    return pd.DataFrame(rows)


def test_resample_then_pivot():
    df = _make_long_df(n_devices=1, n_timestamps=10)
    wide = resample_then_pivot(df, freq="5min")

    assert "var_635" in wide.columns
    assert "var_2782" in wide.columns
    assert "device_serial" in wide.columns


def test_resample_then_pivot_multirate_alignment():
    """Verify that multi-rate variables get properly aligned."""
    df = _make_multirate_long_df(n_devices=1, duration_minutes=30)
    wide = resample_then_pivot(df, freq="1min", max_gap="5min")

    both_present = wide["var_635"].notna() & wide["var_2782"].notna()
    pct_both = both_present.sum() / len(wide)
    # With resampling + ffill, most 1-min bins should have both vars
    assert pct_both > 0.90, f"Only {pct_both:.1%} of rows have both vars aligned"

    # Compute deviation — should be small since both are ~35
    dev = (wide["var_635"] - wide["var_2782"]).dropna()
    assert dev.abs().mean() < 2.0, f"Mean |deviation| too large: {dev.abs().mean():.2f}"


def test_create_windows():
    n = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-12-01", periods=n, freq="5min"),
        "device_serial": "000000000000001",
        "var_635": np.random.randn(n),
        "var_2782": np.random.randn(n),
    })
    windows, metadata = create_windows(df, window="30min", step="5min", feature_cols=["var_635", "var_2782"])

    assert windows.ndim == 3
    assert windows.shape[2] == 2  # 2 features
    assert len(metadata) == windows.shape[0]
    assert "device_serial" in metadata.columns
    assert "window_start" in metadata.columns
