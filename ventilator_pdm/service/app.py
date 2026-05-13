# pdm/service/app.py
"""FastAPI application factory for the PdM production service.

Exposes a REST API and a server-rendered HTML dashboard for:

- Pipeline health and fleet status overview
- Per-device alert history and anomaly-score charts
- Medusa XLSX import and work-order management
- Active-learning label review and retrain controls
- Pilot-data export (ZIP of CSVs)
- On-demand SHAP feature-importance via the XAI API

Use `create_app` to construct and configure the application.  The
``lifespan`` context manager (defined inside `create_app`) handles
database initialisation and APScheduler lifecycle.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ventilator_pdm.service.config import ServiceConfig
from ventilator_pdm.service.database import Database
from ventilator_pdm.service.scheduler import create_scheduler

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
STATIC_DIR = Path(__file__).parent.parent / "static"


def create_app(
    config: ServiceConfig,
    db: Database,
    start_scheduler: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Registers all route handlers, mounts the static-file directory, and
    wires up the Jinja2 template engine.  A ``lifespan`` context manager
    initialises the database and optionally starts the APScheduler
    background scheduler on startup, shutting it down cleanly on exit.

    Args:
        config: Fully-populated `ServiceConfig` for the current
            deployment (server, paths, SMTP, scheduler settings, etc.).
        db: Open `Database` instance that all route handlers use for
            reads and writes.
        start_scheduler: When ``True`` (default), the APScheduler
            instance returned by `create_scheduler` is started on
            application startup.  Set to ``False`` in tests to avoid
            scheduling background jobs.

    Returns:
        A configured `FastAPI` application ready to be served with
        Uvicorn or passed to a test client.
    """

    scheduler = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal scheduler
        # Initialize database on startup
        db.initialize()

        # Register initial model if none exists
        if db.get_active_model() is None:
            db.insert_model_version(
                model_path=str(config.paths.model),
                label_count=0,
                confirmed_count=0,
            )

        if start_scheduler:
            scheduler = create_scheduler(db, config)
            scheduler.start()
            logger.info("Scheduler started")

        yield

        if scheduler:
            scheduler.shutdown()
            logger.info("Scheduler stopped")

    app = FastAPI(title="PdM Service", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Store db and config for routes
    app.state.db = db
    app.state.config = config

    @app.get("/")
    async def health_page(request: Request):
        """Serve the GET / pipeline-health dashboard page.

        Queries the most recent daily-inference and health-check log
        entries, the active model record, and recent failure entries to
        derive an overall pipeline status indicator (``green``, ``yellow``,
        or ``red``).

        Args:
            request: FastAPI/Starlette request object required by
                `Jinja2Templates.TemplateResponse`.

        Returns:
            HTML response rendered from ``health.html`` with context
            keys ``status``, ``last_inference``, ``last_health``,
            ``active_model``, ``recent_errors``, and ``label_count``.
        """
        last_inference = db.get_health_log(job_name="daily_inference", limit=1)
        last_health = db.get_health_log(job_name="health_check", limit=1)
        active_model = db.get_active_model()
        recent_errors = db.get_health_log(status="failure", limit=10)
        label_count = len(db.get_feedback_labels())

        # Determine pipeline status
        status = "green"
        if not last_inference:
            status = "yellow"
        elif last_inference[0]["status"] == "failure":
            status = "red"
        elif last_health and last_health[0]["status"] == "failure":
            status = "yellow"

        return templates.TemplateResponse(request, "health.html", {
            "status": status,
            "last_inference": last_inference[0] if last_inference else None,
            "last_health": last_health[0] if last_health else None,
            "active_model": active_model,
            "recent_errors": recent_errors,
            "label_count": label_count,
            "active_page": "health",
        })

    @app.get("/fleet")
    async def fleet_page(request: Request):
        """Serve the GET /fleet fleet-overview dashboard page.

        Fetches the latest alert row for every known device and renders
        the fleet summary table.

        Args:
            request: FastAPI/Starlette request object.

        Returns:
            HTML response rendered from ``fleet.html`` with context key
            ``devices`` (list of most-recent alert dicts, one per device).
        """
        devices = db.get_latest_device_status()
        return templates.TemplateResponse(request, "fleet.html", {
            "devices": devices,
            "active_page": "fleet",
        })

    @app.get("/device/{serial}")
    async def device_page(request: Request, serial: str):
        """Serve the GET /device/{serial} per-device detail page.

        Looks up the device in ``FLEET_REGISTRY`` and renders its alert
        history and associated CMMS work orders.

        Args:
            request: FastAPI/Starlette request object.
            serial: Elisa 800 serial number from the URL path.

        Returns:
            HTML response rendered from ``device.html`` with context
            keys ``serial``, ``device_info``, ``alerts``, and
            ``work_orders``.

        Raises:
            HTTPException: 404 (plain HTML) if *serial* is not present
                in ``FLEET_REGISTRY``.
        """
        from ventilator_pdm.registry import FLEET_REGISTRY
        if serial not in FLEET_REGISTRY:
            from fastapi.responses import HTMLResponse
            return HTMLResponse("Device not found", status_code=404)

        alerts = db.get_alerts(device_serial=serial, limit=100)
        device_info = FLEET_REGISTRY[serial]
        work_orders = db.get_work_orders(device_reg=device_info["cmms_reg"])

        return templates.TemplateResponse(request, "device.html", {
            "serial": serial,
            "device_info": device_info,
            "alerts": alerts,
            "work_orders": work_orders,
            "active_page": "fleet",
        })

    @app.get("/api/device/{serial}/scores")
    async def device_scores_api(serial: str, days: int = 30):
        """Return GET /api/device/{serial}/scores anomaly-score time series as JSON.

        Fetches the stored anomaly scores and fleet percentiles for the
        requested device over the last *days* calendar days.

        Args:
            serial: Elisa 800 serial number from the URL path.
            days: Look-back window in calendar days. Defaults to 30.

        Returns:
            JSON object with three parallel arrays:
            ``timestamps`` (ISO strings), ``scores`` (float anomaly
            scores), and ``percentiles`` (float fleet percentiles).
        """
        scores = db.get_device_score_history(serial, days=days)
        return {
            "timestamps": [s["timestamp"] for s in scores],
            "scores": [s["anomaly_score"] for s in scores],
            "percentiles": [s["fleet_percentile"] for s in scores],
        }

    # ------------------------------------------------------------------
    # Alert History
    # ------------------------------------------------------------------

    @app.get("/alerts")
    async def alerts_page(request: Request):
        """Serve the GET /alerts alert-history dashboard page.

        Fetches up to 200 most-recent alerts across all devices and
        renders them in a sortable table.

        Args:
            request: FastAPI/Starlette request object.

        Returns:
            HTML response rendered from ``alerts.html`` with context
            key ``alerts`` (list of alert dicts ordered newest-first).
        """
        alerts = db.get_alerts(limit=200)
        return templates.TemplateResponse(request, "alerts.html", {
            "alerts": alerts,
            "active_page": "alerts",
        })

    # ------------------------------------------------------------------
    # Medusa Import
    # ------------------------------------------------------------------

    @app.get("/import")
    async def import_page(request: Request):
        """Serve the GET /import Medusa data-import dashboard page.

        Fetches all stored work orders to display alongside the upload
        form.

        Args:
            request: FastAPI/Starlette request object.

        Returns:
            HTML response rendered from ``import.html`` with context
            key ``work_orders`` (list of work-order dicts).
        """
        work_orders = db.get_work_orders()
        return templates.TemplateResponse(request, "import.html", {
            "work_orders": work_orders,
            "active_page": "import",
        })

    @app.post("/import/upload")
    async def import_upload(file: UploadFile):
        """Parse a POST /import/upload Medusa XLSX file and return a preview.

        Accepts a multipart file upload, writes it to a temporary file,
        parses it with `parse_medusa_excel`, serialises any date fields
        to ISO strings, and returns the parsed records.  The temporary
        file is deleted before returning.

        Args:
            file: Multipart-form file upload containing the Medusa XLSX
                export.

        Returns:
            JSON object with ``filename`` (str) and ``records`` (list of
            work-order dicts with ISO-string date fields).  On parse
            failure returns ``{"error": "<message>"}`` instead.
        """
        import tempfile
        from ventilator_pdm.service.medusa_adapter import parse_medusa_excel

        suffix = Path(file.filename).suffix if file.filename else ".xlsx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        try:
            records = parse_medusa_excel(tmp_path)
        except Exception as exc:
            return {"error": f"Failed to parse file: {exc}"}
        finally:
            tmp_path.unlink(missing_ok=True)

        # Serialise dates to ISO strings for JSON response
        for r in records:
            for key in ("registered_date", "completed_date"):
                if r.get(key) is not None:
                    r[key] = r[key].isoformat()

        return {
            "filename": file.filename,
            "records": records,
        }

    @app.post("/import/confirm")
    async def import_confirm(request: Request):
        """Confirm a POST /import/confirm import and run the label matcher.

        Accepts a JSON body with a ``filename`` field (informational
        only).  Runs `match_work_orders` against all currently unmatched
        work orders in the database and returns summary counts.

        Note:
            The ``/import/upload`` preview step does not persist records
            to the database — the temp file is deleted after preview.
            This endpoint runs the label matcher against any unmatched
            work orders that already exist in the database, regardless
            of how they were originally inserted.

        Args:
            request: FastAPI/Starlette request object.  Body must be a
                JSON object; ``filename`` key is optional.

        Returns:
            JSON object with ``imported`` (total work orders now in DB)
            and ``matches`` (result dict from `match_work_orders`).
        """
        from ventilator_pdm.service.label_matcher import match_work_orders

        body = await request.json()
        filename = body.get("filename", "unknown")

        # Re-parse the uploaded file is not feasible here since the temp file
        # is gone.  Instead we rely on the records already being stored during
        # the preview step.  For robustness, just run the label matcher on any
        # unmatched work orders that exist.
        results = match_work_orders(db)

        return {
            "imported": len(db.get_work_orders()),
            "matches": results,
        }

    # ------------------------------------------------------------------
    # Active Learning
    # ------------------------------------------------------------------

    @app.get("/learning")
    async def learning_page(request: Request):
        """Serve the GET /learning active-learning feedback dashboard page.

        Computes summary statistics over the current label set (total
        labels, confirmed-fault count) and determines whether the label
        threshold for initiating a retrain has been reached.

        Args:
            request: FastAPI/Starlette request object.

        Returns:
            HTML response rendered from ``learning.html`` with context
            keys ``labels``, ``models``, ``total_labels``,
            ``confirmed_count``, and ``retrain_ready`` (bool, True when
            total >= 10 and confirmed >= 1).
        """
        labels = db.get_feedback_labels()
        models = db.get_model_versions()
        total = len(labels)
        confirmed = sum(1 for l in labels if l["event_type"] == "confirmed_fault")
        retrain_ready = total >= 10 and confirmed >= 1
        return templates.TemplateResponse(request, "learning.html", {
            "labels": labels,
            "models": models,
            "total_labels": total,
            "confirmed_count": confirmed,
            "retrain_ready": retrain_ready,
            "active_page": "learning",
        })

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    @app.get("/reports")
    async def reports_page(request: Request):
        """Serve the GET /reports pilot-summary reports dashboard page.

        Aggregates alert and label statistics and renders a summary view
        with model version history and the recent health log.

        Args:
            request: FastAPI/Starlette request object.

        Returns:
            HTML response rendered from ``reports.html`` with context
            keys ``total_alerts``, ``warning_plus``, ``total_labels``,
            ``confirmed_count``, ``models``, and ``health_log``.
        """
        alerts = db.get_alerts(limit=1000)
        labels = db.get_feedback_labels()
        models = db.get_model_versions()
        health_log = db.get_health_log(limit=100)

        total_alerts = len(alerts)
        warning_plus = sum(
            1 for a in alerts if a["alert_level"] in ("warning", "critical")
        )
        total_labels = len(labels)
        confirmed = sum(
            1 for l in labels if l["event_type"] == "confirmed_fault"
        )

        return templates.TemplateResponse(request, "reports.html", {
            "total_alerts": total_alerts,
            "warning_plus": warning_plus,
            "total_labels": total_labels,
            "confirmed_count": confirmed,
            "models": models,
            "health_log": health_log,
            "active_page": "reports",
        })

    @app.get("/api/reports/export")
    async def export_pilot_data():
        """Stream a GET /api/reports/export ZIP archive of all pilot data as CSVs.

        Builds an in-memory ZIP archive containing five CSV files:
        ``alerts.csv``, ``feedback_labels.csv``, ``work_orders.csv``,
        ``health_log.csv``, and ``model_versions.csv``.  Each CSV
        includes a header row derived from the dict keys of the first
        row returned by the corresponding database query.

        Returns:
            `StreamingResponse` with ``Content-Type: application/zip``
            and ``Content-Disposition: attachment;
            filename=pdm_pilot_export.zip``.
        """
        import csv
        import io
        import zipfile

        from fastapi.responses import StreamingResponse

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Export alerts
            alerts = db.get_alerts(limit=10000)
            csv_buf = io.StringIO()
            if alerts:
                writer = csv.DictWriter(csv_buf, fieldnames=alerts[0].keys())
                writer.writeheader()
                writer.writerows(alerts)
            zf.writestr("alerts.csv", csv_buf.getvalue())

            # Export feedback labels
            labels = db.get_feedback_labels()
            csv_buf = io.StringIO()
            if labels:
                writer = csv.DictWriter(csv_buf, fieldnames=labels[0].keys())
                writer.writeheader()
                writer.writerows(labels)
            zf.writestr("feedback_labels.csv", csv_buf.getvalue())

            # Export work orders
            orders = db.get_work_orders()
            csv_buf = io.StringIO()
            if orders:
                writer = csv.DictWriter(csv_buf, fieldnames=orders[0].keys())
                writer.writeheader()
                writer.writerows(orders)
            zf.writestr("work_orders.csv", csv_buf.getvalue())

            # Export health log
            health = db.get_health_log(limit=10000)
            csv_buf = io.StringIO()
            if health:
                writer = csv.DictWriter(csv_buf, fieldnames=health[0].keys())
                writer.writeheader()
                writer.writerows(health)
            zf.writestr("health_log.csv", csv_buf.getvalue())

            # Export model versions
            models = db.get_model_versions()
            csv_buf = io.StringIO()
            if models:
                writer = csv.DictWriter(csv_buf, fieldnames=models[0].keys())
                writer.writeheader()
                writer.writerows(models)
            zf.writestr("model_versions.csv", csv_buf.getvalue())

        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=pdm_pilot_export.zip"
            },
        )

    # ------------------------------------------------------------------
    # XAI API
    # ------------------------------------------------------------------

    @app.get("/api/device/{serial}/xai")
    async def device_xai_api(serial: str):
        """Return GET /api/device/{serial}/xai SHAP feature-importance scores as JSON.

        First checks the ``shap_cache`` table for a previously computed
        result.  On a cache miss, queries QuestDB for the last 7 days of
        telemetry, computes SHAP feature importances via
        `shap_feature_importance`, stores the result in the cache, and
        returns it.

        Args:
            serial: Elisa 800 serial number from the URL path.

        Returns:
            JSON object with ``features`` (list of feature-importance
            dicts ordered by mean absolute SHAP value descending) and ``cached`` (bool
            indicating whether the result came from cache).  If no
            active model is registered returns ``{"error": "No active
            model"}``.  If the device has no telemetry in the last 7
            days returns ``{"features": [], "cached": False}``.
        """
        import json
        from datetime import datetime, timedelta

        cached = db.get_cached_xai(serial)
        if cached:
            return {"features": json.loads(cached["result_json"]), "cached": True}

        # Compute SHAP on-demand (last 7 days)
        from ventilator_pdm.extract import query_questdb
        from ventilator_pdm.features import prepare_features
        from ventilator_pdm.xai import shap_feature_importance
        from ventilator_pdm.models.iforest import IForestModel

        active_model = db.get_active_model()
        if not active_model:
            return {"error": "No active model"}

        model = IForestModel.load(Path(active_model["model_path"]))

        seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        sql = (
            f"SELECT * FROM pdm_medical_device "
            f"WHERE device_serial = '{serial}' "
            f"AND timestamp >= '{seven_days_ago}'"
        )
        df = query_questdb(sql, url=config.questdb.url)
        if df.empty:
            return {"features": [], "cached": False}

        windows, metadata, feature_cols = prepare_features(df)
        importance_df = shap_feature_importance(model, windows, feature_cols)

        result_json = importance_df.to_json(orient="records")
        window_start = metadata["window_start"].min()
        window_end = metadata["window_end"].max()
        db.set_cached_xai(serial, result_json, str(window_start), str(window_end))

        return {
            "features": importance_df.to_dict(orient="records"),
            "cached": False,
        }

    return app
