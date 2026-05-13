"""SQLite database module for PdM service state persistence.

Provides the `Database` class — a thin repository layer on top of SQLite
using the connection-per-operation pattern so that FastAPI async handlers
can call it safely from multiple threads.

Tables managed here:

- ``alert_history`` — one row per device per inference run.
- ``system_health`` — execution log for scheduled jobs.
- ``medusa_work_orders`` — CMMS work orders imported from Medusa XLSX
  exports.
- ``feedback_labels`` — clinician/engineer feedback labels used for
  active learning.
- ``model_versions`` — registry of trained model artefacts with an
  ``active`` flag.
- ``shap_cache`` — cached SHAP feature-importance results per device.
- ``schema_version`` — single-row table recording the DDL version.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# Alert levels ordered by severity for min_level filtering.
_ALERT_LEVELS = ("normal", "warning", "critical")

_SCHEMA_VERSION = 1


class Database:
    """Thread-safe SQLite wrapper using connection-per-operation pattern.

    Each public method opens its own connection, executes, commits, and
    closes -- safe for concurrent use from FastAPI async handlers.
    """

    def __init__(self, db_path: Path | str) -> None:
        """Initialise the database wrapper.

        Args:
            db_path: Filesystem path to the SQLite file.  Passed through
                `Path()` so both strings and `Path` objects are accepted.
                The parent directory must exist; SQLite will create the
                file itself on first connect.
        """
        self.db_path = Path(db_path)

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create all tables if they do not exist (idempotent)."""
        conn = self._connect()
        try:
            conn.executescript(_CREATE_TABLES_SQL)
            # Seed schema_version if empty.
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM schema_version"
            ).fetchone()
            if row["cnt"] == 0:
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (_SCHEMA_VERSION, _now_iso()),
                )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_tables(self) -> list[str]:
        """Return the names of all user-created tables in the database.

        Queries ``sqlite_master`` and excludes internal SQLite tables
        (those whose names start with ``sqlite_``).

        Returns:
            Alphabetically-sorted list of table name strings.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
            return [r["name"] for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # alert_history
    # ------------------------------------------------------------------

    def insert_alert(
        self,
        device_serial: str,
        timestamp: datetime,
        alert_level: str,
        anomaly_score: float,
        fleet_percentile: float,
    ) -> int:
        """Insert a new alert row into ``alert_history``.

        Args:
            device_serial: Elisa 800 serial number identifying the device.
            timestamp: Datetime of the inference window end.
            alert_level: Severity label — one of ``"normal"``,
                ``"warning"``, or ``"critical"``.
            anomaly_score: Raw anomaly score from the model
                (higher = more anomalous).
            fleet_percentile: Position of the score within the current
                fleet distribution, expressed as a fraction [0, 1].

        Returns:
            Auto-incremented primary key (``id``) of the new row.
        """
        conn = self._connect()
        try:
            cur = conn.execute(
                """INSERT INTO alert_history
                   (device_serial, timestamp, alert_level, anomaly_score, fleet_percentile)
                   VALUES (?, ?, ?, ?, ?)""",
                (device_serial, timestamp.isoformat(), alert_level, anomaly_score, fleet_percentile),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_alerts(
        self,
        device_serial: Optional[str] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        min_level: Optional[str] = None,
        limit: int = 100,
        notified_only: bool = False,
    ) -> list[dict]:
        """Query ``alert_history`` with optional filters.

        All filter arguments are optional and combined with ``AND``.
        Results are ordered newest-first.

        Args:
            device_serial: Restrict to a single device serial number.
            from_date: Inclusive lower bound on the alert timestamp
                (date only; time is treated as 00:00:00).
            to_date: Inclusive upper bound on the alert timestamp
                (date only; rows with timestamps up to and including
                end-of-day are returned).
            min_level: Minimum severity to include.  Must be one of
                ``"normal"``, ``"warning"``, or ``"critical"``.  Passing
                ``"warning"`` excludes ``"normal"`` rows.
            limit: Maximum number of rows to return. Defaults to 100.
            notified_only: When ``True``, return only rows where
                ``notified = 1``.

        Returns:
            List of alert dicts (all columns), ordered by
            ``timestamp DESC``, at most *limit* items.

        Raises:
            ValueError: If *min_level* is not one of the three valid
                alert-level strings.
        """
        clauses: list[str] = []
        params: list = []

        if device_serial is not None:
            clauses.append("device_serial = ?")
            params.append(device_serial)
        if from_date is not None:
            clauses.append("timestamp >= ?")
            params.append(from_date.isoformat())
        if to_date is not None:
            clauses.append("timestamp < ?")
            params.append(_next_day(to_date).isoformat())
        if min_level is not None:
            if min_level not in _ALERT_LEVELS:
                raise ValueError(f"Invalid alert level: {min_level!r}. Must be one of {_ALERT_LEVELS}")
            idx = _ALERT_LEVELS.index(min_level)
            allowed = _ALERT_LEVELS[idx:]
            placeholders = ",".join("?" for _ in allowed)
            clauses.append(f"alert_level IN ({placeholders})")
            params.extend(allowed)
        if notified_only:
            clauses.append("notified = 1")

        where = " AND ".join(clauses)
        sql = "SELECT * FROM alert_history"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def get_unnotified_alerts(self) -> list[dict]:
        """Return all alert rows that have not yet triggered an e-mail notification.

        Returns:
            List of alert dicts where ``notified`` is ``False``, ordered
            by ``timestamp DESC``.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM alert_history WHERE notified = 0 ORDER BY timestamp DESC"
            ).fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def mark_alert_notified(self, device_serial: str, timestamp: datetime) -> None:
        """Mark an alert row as notified and record the notification time.

        Sets ``notified = 1`` and ``notified_at`` to the current UTC time
        for the row matching (*device_serial*, *timestamp*).

        Args:
            device_serial: Serial number of the device whose alert is
                being marked.
            timestamp: Exact datetime of the alert row to update.
        """
        conn = self._connect()
        try:
            conn.execute(
                """UPDATE alert_history
                   SET notified = 1, notified_at = ?
                   WHERE device_serial = ? AND timestamp = ?""",
                (_now_iso(), device_serial, timestamp.isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def get_latest_device_status(self) -> list[dict]:
        """Return the most recent alert row for each device."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT a.*
                   FROM alert_history a
                   INNER JOIN (
                       SELECT device_serial, MAX(timestamp) AS max_ts
                       FROM alert_history
                       GROUP BY device_serial
                   ) latest
                   ON a.device_serial = latest.device_serial
                      AND a.timestamp = latest.max_ts
                   ORDER BY a.device_serial"""
            ).fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def get_device_score_history(self, serial: str, days: int = 30) -> list[dict]:
        """Return anomaly score time series for a device over the last *days* calendar days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT timestamp, anomaly_score, fleet_percentile, alert_level
                   FROM alert_history
                   WHERE device_serial = ? AND timestamp >= ?
                   ORDER BY timestamp ASC""",
                (serial, cutoff),
            ).fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # system_health
    # ------------------------------------------------------------------

    def log_health(
        self,
        job_name: str,
        status: str,
        duration_ms: int = 0,
        rows_processed: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """Append a job-execution record to ``system_health``.

        Called at the end of every scheduled job to record whether it
        succeeded or failed, how long it took, and how many rows it
        processed.

        Args:
            job_name: Identifier for the scheduled job, e.g.
                ``"daily_inference"`` or ``"health_check"``.
            status: Outcome string — typically ``"success"`` or
                ``"failure"``.
            duration_ms: Wall-clock execution time in milliseconds.
                Defaults to 0.
            rows_processed: Number of telemetry rows handled by the job.
                Defaults to 0.
            error_message: Exception message or traceback excerpt when
                *status* is ``"failure"``.  ``None`` for successful runs.
        """
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO system_health
                   (timestamp, job_name, status, duration_ms, rows_processed, error_message)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (_now_iso(), job_name, status, duration_ms, rows_processed, error_message),
            )
            conn.commit()
        finally:
            conn.close()

    def get_health_log(
        self,
        job_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """Query ``system_health`` with optional job-name and status filters.

        Args:
            job_name: When provided, restrict results to rows with this
                job name (e.g. ``"daily_inference"``).
            status: When provided, restrict results to rows with this
                status (e.g. ``"failure"``).
            limit: Maximum number of rows to return. Defaults to 10.

        Returns:
            List of health-log dicts ordered by ``timestamp DESC``, at
            most *limit* items.
        """
        clauses: list[str] = []
        params: list = []

        if job_name is not None:
            clauses.append("job_name = ?")
            params.append(job_name)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)

        where = " AND ".join(clauses)
        sql = "SELECT * FROM system_health"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # medusa_work_orders
    # ------------------------------------------------------------------

    def insert_work_order(
        self,
        work_order_id: str,
        work_order_type: str,
        registered_date: Optional[date],
        completed_date: Optional[date],
        fault_description: str,
        technical_description: str,
        device_reg: int,
        serial_number: str,
        import_filename: str,
    ) -> None:
        """Insert a Medusa work order into ``medusa_work_orders`` (ignore duplicates).

        Uses ``INSERT OR IGNORE`` so re-importing the same XLSX file is
        safe; existing rows keyed on *work_order_id* are left unchanged.

        Args:
            work_order_id: Unique CMMS identifier for the work order
                (primary key).
            work_order_type: Category string from the CMMS, e.g.
                ``"corrective"`` or ``"preventive"``.
            registered_date: Date the work order was opened in the CMMS.
                ``None`` if the field was blank in the source file.
            completed_date: Date the work order was closed.  ``None``
                if still open or not recorded.
            fault_description: Free-text description of the reported
                fault from the CMMS.
            technical_description: Free-text engineer notes from the
                CMMS.
            device_reg: Integer CMMS registration number for the device.
            serial_number: Device serial number as recorded in the CMMS
                (may differ in formatting from the HL7 serial).
            import_filename: Name of the XLSX file this row originated
                from, stored for audit purposes.
        """
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO medusa_work_orders
                   (work_order_id, work_order_type, registered_date, completed_date,
                    fault_description, technical_description, device_reg,
                    serial_number, import_filename, imported_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    work_order_id,
                    work_order_type,
                    registered_date.isoformat() if registered_date else None,
                    completed_date.isoformat() if completed_date else None,
                    fault_description,
                    technical_description,
                    device_reg,
                    serial_number,
                    import_filename,
                    _now_iso(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_work_orders(self, device_reg: Optional[int] = None) -> list[dict]:
        """Return work orders from ``medusa_work_orders``, optionally filtered by device.

        Args:
            device_reg: When provided, return only work orders for this
                CMMS registration number.  When ``None`` (default),
                return all work orders.

        Returns:
            List of work-order dicts ordered by ``registered_date DESC``.
        """
        conn = self._connect()
        try:
            if device_reg is not None:
                rows = conn.execute(
                    "SELECT * FROM medusa_work_orders WHERE device_reg = ? ORDER BY registered_date DESC",
                    (device_reg,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM medusa_work_orders ORDER BY registered_date DESC"
                ).fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def get_unmatched_work_orders(self) -> list[dict]:
        """Return work orders that have not yet been matched to an alert.

        Returns:
            List of work-order dicts where ``matched`` is ``False``,
            ordered by ``registered_date DESC``.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM medusa_work_orders WHERE matched = 0 ORDER BY registered_date DESC"
            ).fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def mark_work_order_matched(self, work_order_id: str) -> None:
        """Set ``matched = 1`` on a work order after label-matcher linkage.

        Args:
            work_order_id: Primary key of the work order to update.
        """
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE medusa_work_orders SET matched = 1 WHERE work_order_id = ?",
                (work_order_id,),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # feedback_labels
    # ------------------------------------------------------------------

    def insert_feedback_label(
        self,
        device_serial: str,
        event_date: date,
        event_type: str,
        work_order_id: Optional[str] = None,
        matched_alert_id: Optional[int] = None,
        source: str = "manual",
        notes: str = "",
    ) -> int:
        """Insert a new feedback label into ``feedback_labels``.

        Feedback labels are the ground-truth annotations used by the
        active-learning loop to retrain the anomaly detector.

        Args:
            device_serial: Serial number of the labelled device.
            event_date: Calendar date on which the clinical event or
                fault occurred.
            event_type: Label category, e.g. ``"confirmed_fault"`` or
                ``"false_positive"``.
            work_order_id: CMMS work-order identifier associated with
                this event, if known.
            matched_alert_id: Foreign key to the ``alert_history`` row
                that was matched to this event, if any.
            source: Provenance of the label — ``"manual"`` for
                clinician entry, ``"label_matcher"`` for automatic
                linkage. Defaults to ``"manual"``.
            notes: Free-text annotation from the reviewer.

        Returns:
            Auto-incremented primary key (``id``) of the new row.
        """
        conn = self._connect()
        try:
            cur = conn.execute(
                """INSERT INTO feedback_labels
                   (device_serial, date, event_type, work_order_id,
                    matched_alert_id, notes, created_at, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    device_serial,
                    event_date.isoformat(),
                    event_type,
                    work_order_id,
                    matched_alert_id,
                    notes,
                    _now_iso(),
                    source,
                ),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_feedback_labels(self) -> list[dict]:
        """Return all feedback labels ordered by event date, newest first.

        Returns:
            List of feedback-label dicts (all columns) from
            ``feedback_labels``, ordered by ``date DESC``.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM feedback_labels ORDER BY date DESC"
            ).fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def load_labels(self) -> pd.DataFrame:
        """Return feedback labels as a DataFrame compatible with CMMSFeedbackManager.

        Columns: device_serial, date, event_type, cmms_work_order
        """
        conn = self._connect()
        try:
            df = pd.read_sql_query(
                """SELECT device_serial, date, event_type,
                          work_order_id AS cmms_work_order
                   FROM feedback_labels ORDER BY date DESC""",
                conn,
            )
            return df
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # model_versions
    # ------------------------------------------------------------------

    def insert_model_version(
        self,
        model_path: str,
        label_count: int = 0,
        confirmed_count: int = 0,
        performance_snapshot: Optional[str] = None,
    ) -> int:
        """Register a new trained model and set it as the active model.

        Deactivates all existing model rows (``active = 0``) before
        inserting the new row with ``active = 1``, ensuring at most one
        active model at any time.

        Args:
            model_path: Filesystem path to the serialised model artefact
                (joblib file).
            label_count: Total number of feedback labels available at
                training time. Defaults to 0.
            confirmed_count: Number of confirmed-fault labels used in
                training. Defaults to 0.
            performance_snapshot: Optional JSON string summarising
                evaluation metrics (precision, recall, etc.) for this
                model version.

        Returns:
            Auto-incremented primary key (``version_id``) of the new
            row.
        """
        conn = self._connect()
        try:
            # Deactivate any currently active model.
            conn.execute("UPDATE model_versions SET active = 0 WHERE active = 1")
            cur = conn.execute(
                """INSERT INTO model_versions
                   (trained_at, label_count, confirmed_count,
                    model_path, performance_snapshot, active)
                   VALUES (?, ?, ?, ?, ?, 1)""",
                (_now_iso(), label_count, confirmed_count, model_path, performance_snapshot),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_active_model(self) -> Optional[dict]:
        """Return the currently active model version record.

        Returns:
            Dict of model-version columns for the row where
            ``active = 1``, or ``None`` if no model has been registered
            yet.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE active = 1"
            ).fetchone()
            return _row_to_dict(row) if row else None
        finally:
            conn.close()

    def get_model_versions(self) -> list[dict]:
        """Return all model version records, newest first.

        Returns:
            List of model-version dicts ordered by ``version_id DESC``.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM model_versions ORDER BY version_id DESC"
            ).fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # shap_cache
    # ------------------------------------------------------------------

    def get_cached_xai(self, device_serial: str) -> Optional[dict]:
        """Return the most recent cached SHAP result for a device.

        Args:
            device_serial: Elisa 800 serial number to look up.

        Returns:
            Dict of ``shap_cache`` columns for the latest cached result
            (ordered by ``computed_at DESC``), or ``None`` if no cached
            result exists for this device.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT * FROM shap_cache
                   WHERE device_serial = ?
                   ORDER BY computed_at DESC LIMIT 1""",
                (device_serial,),
            ).fetchone()
            return _row_to_dict(row) if row else None
        finally:
            conn.close()

    def set_cached_xai(
        self,
        device_serial: str,
        result_json: str,
        window_start: str,
        window_end: str,
    ) -> None:
        """Store a SHAP feature-importance result in ``shap_cache``.

        A new row is always inserted; old rows for the same device are
        not deleted.  `get_cached_xai` returns the most recent row, so
        older entries are effectively superseded.

        Args:
            device_serial: Elisa 800 serial number for which SHAP was
                computed.
            result_json: JSON string of the feature-importance records
                (list of dicts with ``feature`` and ``shap_value``
                fields).
            window_start: ISO timestamp of the start of the telemetry
                window used for this computation.
            window_end: ISO timestamp of the end of the telemetry window.
        """
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO shap_cache
                   (device_serial, computed_at, window_start, window_end, result_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (device_serial, _now_iso(), window_start, window_end, result_json),
            )
            conn.commit()
        finally:
            conn.close()


# ======================================================================
# Private helpers
# ======================================================================

def _now_iso() -> str:
    return datetime.now().isoformat()


def _next_day(d: date) -> date:
    """Return the day after *d*, handling month/year boundaries."""
    return d + timedelta(days=1)


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert sqlite3.Row to a plain dict with bool coercion for boolean columns."""
    d = dict(row)
    for key in ("notified", "matched", "active"):
        if key in d and d[key] is not None:
            d[key] = bool(d[key])
    return d


# ======================================================================
# DDL
# ======================================================================

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS alert_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    device_serial   TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL,
    alert_level     TEXT    NOT NULL,
    anomaly_score   REAL,
    fleet_percentile REAL,
    notified        BOOLEAN DEFAULT 0,
    notified_at     TEXT
);

CREATE TABLE IF NOT EXISTS system_health (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    job_name        TEXT    NOT NULL,
    status          TEXT    NOT NULL,
    duration_ms     INTEGER,
    rows_processed  INTEGER DEFAULT 0,
    error_message   TEXT
);

CREATE TABLE IF NOT EXISTS medusa_work_orders (
    work_order_id        TEXT PRIMARY KEY,
    work_order_type      TEXT,
    registered_date      TEXT,
    completed_date       TEXT,
    fault_description    TEXT,
    technical_description TEXT,
    device_reg           INTEGER,
    serial_number        TEXT,
    import_filename      TEXT,
    imported_at          TEXT,
    matched              BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS feedback_labels (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    device_serial    TEXT    NOT NULL,
    date             TEXT    NOT NULL,
    event_type       TEXT    NOT NULL,
    work_order_id    TEXT,
    matched_alert_id INTEGER REFERENCES alert_history(id),
    notes            TEXT    DEFAULT '',
    created_at       TEXT,
    source           TEXT    DEFAULT 'manual'
);

CREATE TABLE IF NOT EXISTS model_versions (
    version_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at          TEXT,
    label_count         INTEGER DEFAULT 0,
    confirmed_count     INTEGER DEFAULT 0,
    model_path          TEXT    NOT NULL,
    performance_snapshot TEXT,
    active              BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS shap_cache (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    device_serial   TEXT    NOT NULL,
    computed_at     TEXT,
    window_start    TEXT,
    window_end      TEXT,
    result_json     TEXT
);

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL,
    applied_at  TEXT    NOT NULL
);
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--db", type=Path, default=Path("data/pdm_service.db"))
    args = parser.parse_args()
    if args.init:
        db = Database(args.db)
        db.initialize()
        print(f"Database initialized at {args.db}")
