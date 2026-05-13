"""APScheduler job definitions for the PdM production service.

Defines the three recurring jobs that make up the automated PdM pipeline:

- :func:`run_health_check` — periodic QuestDB reachability probe.
- :func:`run_daily_inference` — daily anomaly scoring and alert generation
  for the full Elisa 800 fleet.
- :func:`run_label_matching` — periodic CMMS work-order matching and
  automated retrain trigger check.

:func:`create_scheduler` wires these jobs into a
:class:`~apscheduler.schedulers.background.BackgroundScheduler` using cron
and interval triggers read from :class:`~pdm.service.config.ServiceConfig`.
The scheduler runs in a background thread so it does not block the FastAPI
event loop.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from ventilator_pdm.service.config import ServiceConfig
from ventilator_pdm.service.database import Database
from ventilator_pdm.service.alerting import compose_alert_email, compose_error_email, send_email
from ventilator_pdm.service.label_matcher import match_work_orders

logger = logging.getLogger(__name__)


def run_health_check(db: Database, config: ServiceConfig) -> None:
    """Check QuestDB reachability and log the result to the health table.

    Sends an HTTP GET to the QuestDB root endpoint (derived from
    ``config.questdb.url``) with a 10-second timeout.  Records duration and
    outcome (``"success"`` or ``"failure"``) via
    :meth:`~pdm.service.database.Database.log_health`.  Exceptions are
    caught and logged as failures rather than propagated, so a transient
    outage does not crash the scheduler.

    Args:
        db: Open database connection used to persist the health-check result.
        config: Service configuration supplying the QuestDB URL.
    """
    start = time.monotonic()
    try:
        resp = requests.get(
            config.questdb.url.replace("/exec", "/"),
            timeout=10,
        )
        resp.raise_for_status()
        duration = int((time.monotonic() - start) * 1000)
        db.log_health("health_check", "success", duration_ms=duration)
    except Exception as e:
        duration = int((time.monotonic() - start) * 1000)
        db.log_health("health_check", "failure", duration_ms=duration,
                      error_message=str(e))


def run_daily_inference(db: Database, config: ServiceConfig) -> None:
    """Execute the daily fleet inference pipeline.

    Runs the full extract → features → score → alert sequence for the
    previous 24 hours of telemetry:

    1. Fetch the active model path from the database.
    2. Query QuestDB for the last 24 hours of fleet telemetry.
    3. Score the batch with :class:`~pdm.inference.InferenceEngine`.
    4. Generate per-device alerts via the engine's alert logic.
    5. Persist one aggregated alert row per device to SQLite.
    6. Send an email digest for any ``warning`` or ``critical`` devices,
       respecting the configured cooldown period.

    If no data is returned from QuestDB the function returns early after
    logging a zero-row success.  Any unhandled exception is caught, logged,
    and reported via an error notification email; the exception is not
    re-raised so the scheduler continues running.

    Args:
        db: Open database connection used to read the active model and write
            alert rows.
        config: Service configuration supplying QuestDB URL, alert
            thresholds, SMTP settings, and server address for deep links.
    """
    start = time.monotonic()
    try:
        from ventilator_pdm.extract import query_questdb
        from ventilator_pdm.inference import InferenceEngine
        from ventilator_pdm.registry import FLEET_SERIALS
        from ventilator_pdm.variables import FEATURE_IDS

        # 1. Get active model path
        active_model = db.get_active_model()
        if not active_model:
            raise RuntimeError("No active model found in model_versions table")
        model_path = Path(active_model["model_path"])

        # 2. Query last 24h of telemetry (fleet-filtered)
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")
        serials_str = ", ".join(f"'{s}'" for s in FLEET_SERIALS)
        var_ids_str = ", ".join(str(v) for v in FEATURE_IDS)
        sql = (
            f"SELECT * FROM pdm_medical_device "
            f"WHERE timestamp >= '{yesterday}' AND timestamp < '{today}' "
            f"AND device_serial IN ({serials_str}) "
            f"AND variable_id IN ({var_ids_str})"
        )
        df = query_questdb(sql, url=config.questdb.url)

        if df.empty:
            db.log_health("daily_inference", "success", duration_ms=0,
                          rows_processed=0)
            return

        # 3. Score with inference engine
        engine = InferenceEngine(model_path)
        scores_df = engine.score_batch(df)

        # 4. Generate alerts
        alerts_df = engine.generate_alerts(
            scores_df,
            threshold_percentile=config.alerts.threshold_percentile,
            sustained_periods=config.alerts.sustained_periods,
        )

        # 5. Store ONE alert per device per run (aggregated summary)
        now = datetime.now()
        devices_scored = scores_df["device_serial"].unique()
        for serial in devices_scored:
            device_scores = scores_df[scores_df["device_serial"] == serial]
            max_score = device_scores["anomaly_score"].max()
            max_pct = device_scores["fleet_percentile"].max()

            alert_level = "normal"
            if serial in alerts_df["device_serial"].values:
                alert_row = alerts_df[
                    alerts_df["device_serial"] == serial
                ].iloc[0]
                alert_level = alert_row["alert_level"]

            db.insert_alert(
                device_serial=serial,
                timestamp=now,
                alert_level=alert_level,
                anomaly_score=float(max_score),
                fleet_percentile=float(max_pct),
            )

        # 6. Send email for actionable alerts
        actionable = alerts_df[
            alerts_df["alert_level"].isin(["warning", "critical"])
        ]
        if not actionable.empty:
            _send_alert_digest(db, config, actionable)

        duration = int((time.monotonic() - start) * 1000)
        db.log_health("daily_inference", "success",
                      duration_ms=duration, rows_processed=len(df))

    except Exception as e:
        duration = int((time.monotonic() - start) * 1000)
        db.log_health("daily_inference", "failure",
                      duration_ms=duration, error_message=str(e))
        logger.exception("Daily inference failed")
        subject, body = compose_error_email("daily_inference", str(e))
        send_email(config.smtp, subject, body)


def run_label_matching(db: Database, config: ServiceConfig) -> None:
    """Run CMMS work-order matching and check whether a model retrain is due.

    Calls :func:`~pdm.service.label_matcher.match_work_orders` to process
    any new unmatched work orders, then inspects the full feedback-label
    store.  If at least 10 labels have been accumulated and at least one is
    a confirmed fault, the retrain threshold is considered met and
    ``_run_retrain`` is invoked to train and register a new model version.

    Any unhandled exception is caught, logged at ``ERROR`` level, and
    reported via an error notification email; the exception is not
    re-raised.

    Args:
        db: Open database connection used to read feedback labels and work
            orders and to write new labels.
        config: Service configuration supplying QuestDB URL and SMTP
            settings.
    """
    start = time.monotonic()
    try:
        results = match_work_orders(db)

        # Check retrain trigger
        labels = db.get_feedback_labels()
        total = len(labels)
        confirmed = sum(1 for l in labels if l["event_type"] == "confirmed_fault")

        if total >= 10 and confirmed >= 1:
            logger.info("Retrain threshold met: %d labels, %d confirmed", total, confirmed)
            _run_retrain(db, config, total, confirmed)

        duration = int((time.monotonic() - start) * 1000)
        db.log_health("label_matcher", "success",
                      duration_ms=duration, rows_processed=len(results))

    except Exception as e:
        duration = int((time.monotonic() - start) * 1000)
        db.log_health("label_matcher", "failure",
                      duration_ms=duration, error_message=str(e))
        logger.exception("Label matching failed")
        subject, body = compose_error_email("label_matcher", str(e))
        send_email(config.smtp, subject, body)


def _send_alert_digest(db: Database, config: ServiceConfig, alerts_df) -> None:
    """Send email digest for actionable alerts, respecting cooldown."""
    from ventilator_pdm.registry import FLEET_REGISTRY
    cooldown_hours = config.alerts.email_cooldown_hours
    base_url = f"http://{config.server.host}:{config.server.port}"

    alert_records = []
    for _, row in alerts_df.iterrows():
        serial = row["device_serial"]
        # Check cooldown using notified_at (when notification was sent, not alert time)
        recent = db.get_alerts(
            device_serial=serial,
            min_level="warning",
            limit=1,
            notified_only=True,
        )
        if recent and recent[0].get("notified_at"):
            last_notified = datetime.fromisoformat(recent[0]["notified_at"])
            if (datetime.now() - last_notified).total_seconds() < cooldown_hours * 3600:
                continue

        reg_info = FLEET_REGISTRY.get(serial, {})
        alert_records.append({
            "device_serial": serial,
            "device_reg": reg_info.get("cmms_reg", "?"),
            "alert_level": row["alert_level"],
            "fleet_percentile": row.get("fleet_percentile", 0),
            "timestamp": str(row.get("alert_start", datetime.now())),
            "top_features": row.get("top_contributing_features", ""),
        })

    if alert_records:
        subject, body = compose_alert_email(alert_records, base_url=base_url)
        if send_email(config.smtp, subject, body):
            for a in alert_records:
                # Look up the actual latest alert timestamp for this device
                latest = db.get_alerts(device_serial=a["device_serial"], limit=1)
                if latest:
                    alert_ts = datetime.fromisoformat(latest[0]["timestamp"])
                    db.mark_alert_notified(a["device_serial"], alert_ts)


def _run_retrain(db: Database, config: ServiceConfig, label_count: int, confirmed_count: int) -> None:
    """Retrain IForest model with labels from SQLite (not JSONL)."""
    from ventilator_pdm.extract import query_questdb
    from ventilator_pdm.features import prepare_features
    from ventilator_pdm.models.iforest import IForestModel

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"outputs/models/iforest/iforest_{timestamp}.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract full fleet data for training (IForest is unsupervised — labels
    # are tracked in model_versions for provenance but not used in training)
    from ventilator_pdm.registry import FLEET_SERIALS
    serials_str = ", ".join(f"'{s}'" for s in FLEET_SERIALS)
    sql = f"SELECT * FROM pdm_medical_device WHERE device_serial IN ({serials_str})"
    df = query_questdb(sql, url=config.questdb.url)

    # 3. Feature pipeline
    windows, metadata, feature_cols = prepare_features(df)

    # 4. Train new IForest (same hyperparameters)
    model = IForestModel(n_estimators=200, contamination=0.01, random_state=42)
    model.fit(windows)
    model.save(output_path)

    # 5. Register new model version in SQLite
    db.insert_model_version(
        model_path=str(output_path),
        label_count=label_count,
        confirmed_count=confirmed_count,
    )

    logger.info("Model retrained: %s (%d labels, %d confirmed)",
                output_path, label_count, confirmed_count)

    # Send success notification (not error email)
    subject = f"PdM Model Retrained — {label_count} labels"
    body = (
        f"Model retrained successfully.\n"
        f"Labels: {label_count} ({confirmed_count} confirmed faults)\n"
        f"Model path: {output_path}\n"
        f"Time: {datetime.now().isoformat()}\n"
    )
    send_email(config.smtp, subject, body)


def create_scheduler(db: Database, config: ServiceConfig) -> BackgroundScheduler:
    """Create and configure the APScheduler background scheduler.

    Registers three jobs:

    - ``daily_inference``: runs :func:`run_daily_inference` on the cron
      schedule defined by ``config.scheduler.inference_cron``.
    - ``label_matcher``: runs :func:`run_label_matching` on the cron
      schedule defined by ``config.scheduler.label_match_cron``.
    - ``health_check``: runs :func:`run_health_check` at the interval
      defined by ``config.scheduler.health_check_interval_minutes``.

    Cron expressions are expected in standard five-field format
    ``"min hour day month dow"``.  The scheduler is not started here;
    the caller is responsible for calling :meth:`~BackgroundScheduler.start`.

    Args:
        db: Database instance passed as an argument to each scheduled job.
        config: Service configuration providing cron expressions, interval,
            and timezone for all jobs.

    Returns:
        A configured but not-yet-started
        :class:`~apscheduler.schedulers.background.BackgroundScheduler`.
    """
    scheduler = BackgroundScheduler(timezone=config.scheduler.timezone)

    # Parse cron expressions
    inf_parts = config.scheduler.inference_cron.split()
    lm_parts = config.scheduler.label_match_cron.split()

    scheduler.add_job(
        run_daily_inference,
        CronTrigger(
            minute=inf_parts[0], hour=inf_parts[1],
            day=inf_parts[2], month=inf_parts[3],
            day_of_week=inf_parts[4],
            timezone=config.scheduler.timezone,
        ),
        args=[db, config],
        id="daily_inference",
        name="Daily fleet inference",
    )

    scheduler.add_job(
        run_label_matching,
        CronTrigger(
            minute=lm_parts[0], hour=lm_parts[1],
            day=lm_parts[2], month=lm_parts[3],
            day_of_week=lm_parts[4],
            timezone=config.scheduler.timezone,
        ),
        args=[db, config],
        id="label_matcher",
        name="Label matching + retrain check",
    )

    scheduler.add_job(
        run_health_check,
        IntervalTrigger(minutes=config.scheduler.health_check_interval_minutes),
        args=[db, config],
        id="health_check",
        name="Health check",
    )

    return scheduler
