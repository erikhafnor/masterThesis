"""Known-failure-centric evaluation for the Elisa 800 PdM pipeline.

The Elisa 800 fleet at Helse Stavanger has four confirmed O2-sensor failures
recorded in the CMMS (``pdm.registry.KNOWN_FAILURES``).  This module provides
functions that evaluate any anomaly-scoring model against those four events
using the following protocol:

1. **Window extraction** — for each failure, slice the scored telemetry to
   ``[event - pre_days, event + post_days]`` (default 14 days before, 2 after).
2. **Fleet-percentile rank** — at every timestamp express each device's score
   as a percentile across the whole fleet, making models with different score
   scales directly comparable.
3. **Detection lead time** — find the earliest run of ``sustained_periods``
   consecutive score-points that exceed ``threshold_percentile`` in the
   pre-event window; report how many days before the failure that run begins.
4. **Model comparison** — aggregate per-failure results from multiple models
   into a single summary table (detections out of 4, mean lead time).

Primary metrics reported
------------------------
* ``detected`` (bool): Whether a sustained exceedance was found before the event.
* ``lead_time_days`` (float): Days of advance warning; ``None`` if not detected.
* ``max_percentile_pre`` (float): Peak fleet-percentile rank in the pre-event
  window; useful for ranking partially-detecting models.
* ``n_score_points`` (int): Number of scored timesteps in the evaluation window
  (data-coverage sanity check).

Expected DataFrame shapes
--------------------------
* **scores_df** (input to most functions): columns ``device_serial`` (str),
  ``timestamp`` (datetime or UTC-aware Timestamp), ``anomaly_score`` (float).
  One row per scored sample.
* **event_windows** (output of :func:`extract_event_windows`): same columns
  plus ``days_to_event`` (float), ``cmms_reg`` (int), ``event_date``
  (Timestamp), and ``description`` (str).
* **evaluation result** (output of :func:`evaluate_model`): one row per known
  failure; columns listed under ``rows.append(...)`` in that function.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from ventilator_pdm.registry import KNOWN_FAILURES

logger = logging.getLogger(__name__)


def extract_event_windows(
    scores_df: pd.DataFrame,
    pre_days: int = 14,
    post_days: int = 2,
) -> dict[str, pd.DataFrame]:
    """Extract anomaly score windows around each known failure.

    Args:
        scores_df: DataFrame with columns [device_serial, timestamp, anomaly_score].
        pre_days: Days before event to include.
        post_days: Days after event to include.

    Returns:
        Dict keyed by telemetry_serial → score trajectory DataFrame.
    """
    scores_df = scores_df.copy()
    scores_df["timestamp"] = pd.to_datetime(scores_df["timestamp"], utc=True)

    event_windows = {}
    for failure in KNOWN_FAILURES:
        serial = failure["telemetry_serial"]
        event_date = pd.Timestamp(failure["date"], tz="UTC")
        t_start = event_date - timedelta(days=pre_days)
        t_end = event_date + timedelta(days=post_days)

        device_scores = scores_df[
            (scores_df["device_serial"] == serial)
            & (scores_df["timestamp"] >= t_start)
            & (scores_df["timestamp"] <= t_end)
        ].copy()

        device_scores["days_to_event"] = (
            (device_scores["timestamp"] - event_date).dt.total_seconds() / 86400
        )
        device_scores["cmms_reg"] = failure["cmms_reg"]
        device_scores["event_date"] = event_date
        device_scores["description"] = failure["description"]

        event_windows[serial] = device_scores
        logger.info(
            "Reg %d: %d score points in [%s, %s]",
            failure["cmms_reg"], len(device_scores),
            t_start.date(), t_end.date(),
        )

    return event_windows


def fleet_percentile_rank(
    scores_df: pd.DataFrame,
    score_col: str = "anomaly_score",
) -> pd.DataFrame:
    """Add a ``fleet_percentile`` column expressing each score as a fleet-wide rank.

    Groups by ``timestamp`` and ranks values within each group using pandas
    ``rank(pct=True)``, so the result is in ``[0.0, 1.0]`` where 1.0 means
    the highest anomaly score among all devices at that timestamp.

    Args:
        scores_df: DataFrame with at least columns ``timestamp`` and the
            column named by ``score_col``.  One row per (device, timestamp)
            scored sample.
        score_col: Name of the column containing raw anomaly scores.
            Defaults to ``"anomaly_score"``.

    Returns:
        A copy of ``scores_df`` with an additional ``fleet_percentile`` column
        (float, range ``[0.0, 1.0]``).
    """
    scores_df = scores_df.copy()
    scores_df["fleet_percentile"] = (
        scores_df.groupby("timestamp")[score_col].rank(pct=True)
    )
    return scores_df


def detection_lead_time(
    event_scores: pd.DataFrame,
    threshold_percentile: float = 0.95,
    sustained_periods: int = 3,
    score_col: str = "anomaly_score",
) -> float | None:
    """Return how many days before the failure the model first triggered a sustained alert.

    Scans the pre-event portion of ``event_scores`` for the earliest run of
    at least ``sustained_periods`` consecutive timesteps whose
    ``fleet_percentile`` is at or above ``threshold_percentile``.  The lead
    time is the number of days between that first detection point and the
    event date.

    A positive return value means the detection occurred before the failure
    (the typical desired case).  ``None`` means no sustained exceedance was
    found in the pre-event window, i.e. the model missed the event.

    Args:
        event_scores: Per-device score trajectory DataFrame for one failure
            event, as returned by :func:`extract_event_windows`.  Must
            contain columns ``timestamp`` (UTC-aware), ``event_date``
            (UTC-aware Timestamp scalar in the first row), and
            ``fleet_percentile`` (float, ``[0.0, 1.0]``).
        threshold_percentile: Fleet-percentile threshold above which a
            timestep is considered anomalous.  Defaults to ``0.95``
            (top 5 % of the fleet).
        sustained_periods: Minimum number of *consecutive* timesteps that
            must all exceed the threshold to count as a detection, reducing
            false positives from transient spikes.  Defaults to ``3``.
        score_col: Kept for API consistency; not used in the current
            implementation (detection uses ``fleet_percentile``).

    Returns:
        Lead time in days (positive = detected before failure) if a sustained
        exceedance was found, or ``None`` if the event was not detected.
    """
    if event_scores.empty:
        return None

    event_date = event_scores["event_date"].iloc[0]
    pre_event = event_scores[event_scores["timestamp"] < event_date].sort_values("timestamp")

    if pre_event.empty or "fleet_percentile" not in pre_event.columns:
        return None

    above = pre_event["fleet_percentile"] >= threshold_percentile

    # Find first run of `sustained_periods` consecutive True values
    runs = above.astype(int).groupby((above != above.shift()).cumsum())
    for _, run in runs:
        if run.iloc[0] and len(run) >= sustained_periods:
            first_detection = pre_event.loc[run.index[0], "timestamp"]
            lead_time = (event_date - first_detection).total_seconds() / 86400
            return lead_time

    return None


def evaluate_model(
    scores_df: pd.DataFrame,
    model_name: str,
    score_col: str = "anomaly_score",
    threshold_percentile: float = 0.95,
    sustained_periods: int = 3,
) -> pd.DataFrame:
    """Evaluate a single anomaly-scoring model against all four known O2-sensor failures.

    Orchestrates the full evaluation protocol: fleet-percentile ranking,
    event-window extraction, and lead-time detection for each failure.
    Logs a one-line summary (detections / 4, mean lead time) at INFO level.

    Args:
        scores_df: Model output DataFrame with columns ``device_serial`` (str),
            ``timestamp`` (datetime-like), and the column named by
            ``score_col`` (float).  All fleet devices and their full scored
            history should be present so that fleet-percentile ranks are
            computed correctly.
        model_name: Human-readable model identifier included as a ``model``
            column in the returned DataFrame (e.g. ``"Autoencoder v3"``).
        score_col: Name of the anomaly-score column in ``scores_df``.
            Defaults to ``"anomaly_score"``.
        threshold_percentile: Fleet-percentile threshold passed to
            :func:`detection_lead_time`.  Defaults to ``0.95``.
        sustained_periods: Consecutive-period requirement passed to
            :func:`detection_lead_time`.  Defaults to ``3``.

    Returns:
        DataFrame with one row per known failure and columns:
        ``model``, ``cmms_reg``, ``event_date``, ``description``,
        ``detected`` (bool), ``lead_time_days`` (float or ``None``),
        ``max_percentile_pre`` (float or ``None``), ``n_score_points`` (int).
    """
    scores_df = fleet_percentile_rank(scores_df, score_col)
    event_windows = extract_event_windows(scores_df)

    rows = []
    for failure in KNOWN_FAILURES:
        serial = failure["telemetry_serial"]
        window = event_windows.get(serial, pd.DataFrame())

        if window.empty:
            rows.append({
                "model": model_name,
                "cmms_reg": failure["cmms_reg"],
                "event_date": failure["date"],
                "description": failure["description"],
                "detected": False,
                "lead_time_days": None,
                "max_percentile_pre": None,
                "n_score_points": 0,
            })
            continue

        # Merge fleet percentile
        window = window.merge(
            scores_df[["device_serial", "timestamp", "fleet_percentile"]],
            on=["device_serial", "timestamp"],
            how="left",
            suffixes=("", "_fleet"),
        )
        if "fleet_percentile_fleet" in window.columns:
            window["fleet_percentile"] = window["fleet_percentile_fleet"]

        lead = detection_lead_time(
            window,
            threshold_percentile=threshold_percentile,
            sustained_periods=sustained_periods,
            score_col=score_col,
        )

        event_date = pd.Timestamp(failure["date"], tz="UTC")
        pre = window[window["timestamp"] < event_date]
        max_pct = pre["fleet_percentile"].max() if not pre.empty and "fleet_percentile" in pre.columns else None

        rows.append({
            "model": model_name,
            "cmms_reg": failure["cmms_reg"],
            "event_date": failure["date"],
            "description": failure["description"],
            "detected": lead is not None,
            "lead_time_days": lead,
            "max_percentile_pre": max_pct,
            "n_score_points": len(window),
        })

    result = pd.DataFrame(rows)
    n_detected = result["detected"].sum()
    mean_lead = result.loc[result["detected"], "lead_time_days"].mean()
    logger.info(
        "%s: detected %d/4 failures, mean lead time %.1f days",
        model_name, n_detected, mean_lead if not np.isnan(mean_lead) else 0,
    )
    return result


def model_comparison_table(results: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine per-model evaluation results into a single comparison table.

    Concatenates the DataFrames returned by :func:`evaluate_model` for
    different models, then groups by ``model`` to produce one summary row per
    model.  The ``events_detected`` column is formatted as ``"N/4"`` for
    readability.

    Args:
        results: List of DataFrames, each the output of :func:`evaluate_model`
            for one model.  All DataFrames must share the same schema.

    Returns:
        DataFrame with columns ``model`` (str), ``events_detected`` (str,
        e.g. ``"3/4"``), and ``mean_lead_time`` (float, mean of non-``None``
        lead-time values across detected failures).
    """
    all_results = pd.concat(results, ignore_index=True)

    summary = (
        all_results.groupby("model")
        .agg(
            events_detected=("detected", "sum"),
            mean_lead_time=("lead_time_days", "mean"),
        )
        .reset_index()
    )
    summary["events_detected"] = summary["events_detected"].astype(int).astype(str) + "/4"
    return summary
