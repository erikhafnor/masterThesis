"""PV (scheduled maintenance) negative control evaluation.

This module provides a *negative control* check that is the mirror image of
fault-based evaluation in :mod:`pdm.evaluation`.

``pdm.evaluation`` asks: does the model detect *known faults* early?
``pdm.evaluation_pv`` asks: does the model stay *silent* before *scheduled
maintenance* (PV) events?

Background
----------
Helse Stavanger conducted fleet-wide preventive-maintenance inspections of all
Elisa 800 ventilators in December 2025.  Because PV events are scheduled
independently of device condition, anomaly scores in the days preceding a PV
date should be indistinguishable from fleet baseline.  Elevated pre-PV scores
are a red flag: they indicate the model has learned to react to routine
maintenance artefacts (e.g. sensor re-calibration drift) rather than genuine
device degradation.

Typical usage
-------------
::

    from ventilator_pdm.evaluation_pv import evaluate_pv_events

    result = evaluate_pv_events(scores_df)
    n_bad = result["elevated"].sum()
    print(f"{n_bad}/{len(result)} PV events showed spurious pre-event elevation")

The :func:`evaluate_pv_events` function iterates over :data:`pdm.registry.PV_EVENTS`,
computes the mean anomaly score in a configurable pre-event window for each
device, and flags events where that mean lies more than 2 standard deviations
above the fleet-wide baseline.  Ideally zero events are flagged.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from ventilator_pdm.registry import PV_EVENTS

logger = logging.getLogger(__name__)


def evaluate_pv_events(
    scores_df: pd.DataFrame,
    score_col: str = "anomaly_score",
    pre_days: int = 7,
    post_days: int = 1,
    threshold_percentile: float = 0.95,
) -> pd.DataFrame:
    """Check whether PV events show pre-event anomaly elevation.

    Iterates over every entry in :data:`pdm.registry.PV_EVENTS` and, for each
    device, extracts the anomaly scores in the ``pre_days``-day window
    immediately preceding the scheduled maintenance date.  The window mean is
    compared against the fleet-wide mean and standard deviation to produce a
    z-score.  A z-score above 2.0 (>2 sigma) is flagged as *elevated*.

    A well-calibrated model should produce **zero elevated events**: scheduled
    maintenance is condition-independent, so pre-PV behaviour should be
    statistically identical to fleet baseline.

    Args:
        scores_df: Long-format DataFrame of anomaly scores.  Must contain
            columns ``device_serial``, ``timestamp``, and the column named by
            ``score_col``.  Timestamps may be timezone-naive or tz-aware; they
            are coerced to UTC internally.
        score_col: Name of the column in ``scores_df`` that holds the anomaly
            score.  Defaults to ``"anomaly_score"``.
        pre_days: Length of the look-back window (in days) before each PV date
            used to compute the pre-event mean score.  Defaults to 7.
        post_days: Reserved for future use; currently unused in the
            computation.  Defaults to 1.
        threshold_percentile: Reserved for future use; the current
            implementation uses a fixed 2-sigma z-score threshold rather than
            a percentile cutoff.  Defaults to 0.95.

    Returns:
        DataFrame with one row per PV event and the following columns:

        - ``cmms_reg`` (int): CMMS registration number of the device.
        - ``pv_date`` (str): Date of the scheduled maintenance event.
        - ``mean_score_pre`` (float or None): Mean anomaly score in the
            pre-event window.  ``None`` when no telemetry data is available.
        - ``zscore_vs_fleet`` (float or None): Z-score of ``mean_score_pre``
            relative to the fleet-wide mean and standard deviation.
        - ``elevated`` (bool or None): ``True`` when
            ``zscore_vs_fleet > 2.0``, indicating suspicious pre-event
            elevation.  ``None`` when data is missing.
        - ``n_points`` (int): Number of score observations in the pre-event
            window.

    Note:
        Devices with no telemetry in the pre-event window are included in the
        result with ``None`` values rather than being silently dropped, so
        callers can distinguish *no data* from *no elevation*.
    """
    scores_df = scores_df.copy()
    scores_df["timestamp"] = pd.to_datetime(scores_df["timestamp"], utc=True)

    # Fleet-wide baseline statistics
    fleet_mean = scores_df[score_col].mean()
    fleet_std = scores_df[score_col].std()

    rows = []
    for pv in PV_EVENTS:
        serial = pv["telemetry_serial"]
        event_date = pd.Timestamp(pv["date"], tz="UTC")
        t_start = event_date - timedelta(days=pre_days)

        pre_scores = scores_df[
            (scores_df["device_serial"] == serial)
            & (scores_df["timestamp"] >= t_start)
            & (scores_df["timestamp"] < event_date)
        ]

        if pre_scores.empty:
            rows.append({
                "cmms_reg": pv["cmms_reg"],
                "pv_date": pv["date"],
                "mean_score_pre": None,
                "zscore_vs_fleet": None,
                "elevated": None,
                "n_points": 0,
            })
            continue

        mean_pre = pre_scores[score_col].mean()
        z = (mean_pre - fleet_mean) / fleet_std if fleet_std > 0 else 0

        rows.append({
            "cmms_reg": pv["cmms_reg"],
            "pv_date": pv["date"],
            "mean_score_pre": mean_pre,
            "zscore_vs_fleet": z,
            "elevated": z > 2.0,  # >2σ above fleet mean
            "n_points": len(pre_scores),
        })

    result = pd.DataFrame(rows)
    n_elevated = result["elevated"].sum() if "elevated" in result.columns else 0
    logger.info(
        "PV evaluation: %d/%d events with pre-event elevation (want: 0)",
        n_elevated, len(PV_EVENTS),
    )
    return result
