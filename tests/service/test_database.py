import pytest
from datetime import datetime, date
from pathlib import Path
from ventilator_pdm.service.database import Database


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    database = Database(db_path)
    database.initialize()
    return database


def test_initialize_creates_tables(db):
    tables = db.list_tables()
    assert "alert_history" in tables
    assert "system_health" in tables
    assert "medusa_work_orders" in tables
    assert "feedback_labels" in tables
    assert "model_versions" in tables
    assert "schema_version" in tables
    assert "shap_cache" in tables


def test_initialize_is_idempotent(db):
    """Calling initialize twice should not raise."""
    db.initialize()
    tables = db.list_tables()
    assert len(tables) == 7


def test_insert_and_query_alert(db):
    db.insert_alert(
        device_serial="000000008204506",
        timestamp=datetime(2026, 4, 1, 6, 0),
        alert_level="warning",
        anomaly_score=-0.15,
        fleet_percentile=0.97,
    )
    alerts = db.get_alerts(limit=10)
    assert len(alerts) == 1
    assert alerts[0]["device_serial"] == "000000008204506"
    assert alerts[0]["alert_level"] == "warning"
    assert alerts[0]["notified"] is False


def test_insert_and_query_system_health(db):
    db.log_health(
        job_name="daily_inference",
        status="success",
        duration_ms=45000,
        rows_processed=51000,
    )
    entries = db.get_health_log(limit=5)
    assert len(entries) == 1
    assert entries[0]["job_name"] == "daily_inference"
    assert entries[0]["status"] == "success"


def test_get_health_log_filter_by_job_name(db):
    db.log_health("daily_inference", "success", duration_ms=100)
    db.log_health("health_check", "success", duration_ms=50)
    db.log_health("health_check", "failure", duration_ms=10, error_message="refused")

    entries = db.get_health_log(job_name="health_check", limit=5)
    assert len(entries) == 2
    assert all(e["job_name"] == "health_check" for e in entries)


def test_get_health_log_filter_by_status(db):
    db.log_health("daily_inference", "success", duration_ms=100)
    db.log_health("health_check", "failure", duration_ms=10, error_message="err")

    entries = db.get_health_log(status="failure", limit=10)
    assert len(entries) == 1
    assert entries[0]["status"] == "failure"


def test_insert_work_order(db):
    db.insert_work_order(
        work_order_id="AO-12345",
        work_order_type="corrective",
        registered_date=date(2026, 1, 9),
        completed_date=date(2026, 1, 10),
        fault_description="O2 sensor feil",
        technical_description="Byttet O2 sensor",
        device_reg=18839,
        serial_number="8204570",
        import_filename="medusa_export.xlsx",
    )
    orders = db.get_work_orders()
    assert len(orders) == 1
    assert orders[0]["work_order_id"] == "AO-12345"
    assert orders[0]["matched"] is False


def test_get_work_orders_filter_by_device_reg(db):
    db.insert_work_order("AO-1", "corrective", date(2026, 1, 9), None,
                         "Feil", "", 18839, "8204570", "test.xlsx")
    db.insert_work_order("AO-2", "preventive", date(2025, 12, 8), None,
                         "PV", "", 18817, "8204506", "test.xlsx")
    orders = db.get_work_orders(device_reg=18839)
    assert len(orders) == 1
    assert orders[0]["work_order_id"] == "AO-1"


def test_insert_and_query_feedback_label(db):
    db.insert_feedback_label(
        device_serial="000000008204570",
        event_date=date(2026, 1, 9),
        event_type="confirmed_fault",
        work_order_id="AO-12345",
        matched_alert_id=None,
        source="auto",
    )
    labels = db.get_feedback_labels()
    assert len(labels) == 1
    assert labels[0]["event_type"] == "confirmed_fault"


def test_load_labels_compat_format(db):
    """load_labels() returns DataFrame compatible with CMMSFeedbackManager."""
    db.insert_feedback_label(
        device_serial="000000008204570",
        event_date=date(2026, 1, 9),
        event_type="confirmed_fault",
        work_order_id="AO-12345",
        source="auto",
    )
    df = db.load_labels()
    for col in ["device_serial", "date", "event_type", "cmms_work_order"]:
        assert col in df.columns
    assert len(df) == 1


def test_model_versions(db):
    db.insert_model_version(
        model_path="outputs/models/iforest/iforest.joblib",
        label_count=0,
        confirmed_count=0,
    )
    active = db.get_active_model()
    assert active is not None
    assert active["model_path"] == "outputs/models/iforest/iforest.joblib"
    assert active["active"] is True


def test_model_version_swap(db):
    db.insert_model_version(
        model_path="outputs/models/iforest/iforest_v1.joblib",
        label_count=0,
        confirmed_count=0,
    )
    db.insert_model_version(
        model_path="outputs/models/iforest/iforest_v2.joblib",
        label_count=12,
        confirmed_count=2,
    )
    active = db.get_active_model()
    assert active["model_path"] == "outputs/models/iforest/iforest_v2.joblib"

    # Previous model should be inactive
    all_versions = db.get_model_versions()
    assert len(all_versions) == 2
    assert all_versions[1]["active"] is False


def test_get_alerts_for_device(db):
    db.insert_alert("serial_a", datetime(2026, 4, 1), "warning", -0.1, 0.96)
    db.insert_alert("serial_b", datetime(2026, 4, 1), "normal", -0.3, 0.50)
    db.insert_alert("serial_a", datetime(2026, 4, 2), "critical", -0.05, 0.99)

    alerts = db.get_alerts(device_serial="serial_a")
    assert len(alerts) == 2
    assert all(a["device_serial"] == "serial_a" for a in alerts)


def test_get_alerts_filter_by_date_range(db):
    db.insert_alert("serial_a", datetime(2026, 4, 1), "warning", -0.1, 0.96)
    db.insert_alert("serial_a", datetime(2026, 4, 5), "critical", -0.05, 0.99)
    db.insert_alert("serial_a", datetime(2026, 4, 10), "warning", -0.12, 0.97)

    alerts = db.get_alerts(
        device_serial="serial_a",
        from_date=date(2026, 4, 3),
        to_date=date(2026, 4, 7),
    )
    assert len(alerts) == 1
    assert "2026-04-05" in alerts[0]["timestamp"]


def test_get_alerts_filter_by_min_level(db):
    db.insert_alert("serial_a", datetime(2026, 4, 1), "normal", -0.3, 0.50)
    db.insert_alert("serial_a", datetime(2026, 4, 1), "warning", -0.1, 0.96)
    db.insert_alert("serial_a", datetime(2026, 4, 1), "critical", -0.05, 0.99)

    alerts = db.get_alerts(min_level="warning")
    assert len(alerts) == 2
    assert all(a["alert_level"] in ("warning", "critical") for a in alerts)


def test_get_unnotified_alerts(db):
    db.insert_alert("serial_a", datetime(2026, 4, 1), "warning", -0.1, 0.96)
    db.insert_alert("serial_a", datetime(2026, 4, 2), "critical", -0.05, 0.99)
    db.mark_alert_notified(device_serial="serial_a", timestamp=datetime(2026, 4, 1))

    unnotified = db.get_unnotified_alerts()
    assert len(unnotified) == 1
    assert unnotified[0]["timestamp"] == datetime(2026, 4, 2).isoformat()


def test_get_latest_device_status(db):
    db.insert_alert("serial_a", datetime(2026, 4, 1), "warning", -0.1, 0.96)
    db.insert_alert("serial_a", datetime(2026, 4, 5), "critical", -0.05, 0.99)
    db.insert_alert("serial_b", datetime(2026, 4, 3), "normal", -0.3, 0.50)

    latest = db.get_latest_device_status()
    assert len(latest) == 2
    by_serial = {r["device_serial"]: r for r in latest}
    assert "2026-04-05" in by_serial["serial_a"]["timestamp"]
    assert by_serial["serial_a"]["alert_level"] == "critical"
    assert by_serial["serial_b"]["alert_level"] == "normal"


def test_get_device_score_history(db):
    now = datetime.now()
    db.insert_alert("serial_a", now, "warning", -0.1, 0.96)
    db.insert_alert("serial_a", datetime(2020, 1, 1), "normal", -0.5, 0.30)

    history = db.get_device_score_history("serial_a", days=30)
    assert len(history) == 1
    assert history[0]["anomaly_score"] == -0.1


def test_get_unmatched_work_orders(db):
    db.insert_work_order("AO-1", "corrective", date(2026, 1, 9), None,
                         "Feil", "", 18839, "8204570", "test.xlsx")
    db.insert_work_order("AO-2", "preventive", date(2025, 12, 8), None,
                         "PV", "", 18817, "8204506", "test.xlsx")
    db.mark_work_order_matched("AO-1")

    unmatched = db.get_unmatched_work_orders()
    assert len(unmatched) == 1
    assert unmatched[0]["work_order_id"] == "AO-2"


def test_xai_cache_round_trip(db):
    db.set_cached_xai("serial_a", '{"features": [1, 2]}', "2026-04-01", "2026-04-07")

    cached = db.get_cached_xai("serial_a")
    assert cached is not None
    assert cached["device_serial"] == "serial_a"
    assert cached["result_json"] == '{"features": [1, 2]}'

    assert db.get_cached_xai("nonexistent") is None


def test_get_alerts_invalid_min_level(db):
    with pytest.raises(ValueError, match="Invalid alert level"):
        db.get_alerts(min_level="unknown")
