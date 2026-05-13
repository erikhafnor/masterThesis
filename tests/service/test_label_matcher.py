import pytest
from datetime import datetime, date
from ventilator_pdm.service.database import Database
from ventilator_pdm.service.label_matcher import match_work_orders


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    return database


def test_match_corrective_fault_to_alert(db):
    """Corrective work order with fault keyword matches nearby alert."""
    db.insert_alert(
        device_serial="000000008204570",
        timestamp=datetime(2026, 1, 7, 6, 0),
        alert_level="warning",
        anomaly_score=-0.1,
        fleet_percentile=0.97,
    )
    db.insert_work_order(
        work_order_id="AO-001",
        work_order_type="corrective",
        registered_date=date(2026, 1, 9),
        completed_date=date(2026, 1, 10),
        fault_description="O2 sensor feil #259",
        technical_description="Byttet sensor",
        device_reg=18839,
        serial_number="8204570",
        import_filename="test.xlsx",
    )

    results = match_work_orders(db)
    assert len(results) == 1
    assert results[0]["event_type"] == "confirmed_fault"


def test_match_preventive_to_scheduled_pv(db):
    db.insert_alert(
        device_serial="000000008204506",
        timestamp=datetime(2025, 12, 7, 6, 0),
        alert_level="warning",
        anomaly_score=-0.2,
        fleet_percentile=0.96,
    )
    db.insert_work_order(
        work_order_id="AO-002",
        work_order_type="preventive",
        registered_date=date(2025, 12, 8),
        completed_date=date(2025, 12, 8),
        fault_description="PV kontroll",
        technical_description="Forebyggende vedlikehold",
        device_reg=18817,
        serial_number="8204506",
        import_filename="test.xlsx",
    )

    results = match_work_orders(db)
    assert len(results) == 1
    assert results[0]["event_type"] == "scheduled_pv"


def test_no_match_when_no_nearby_alert(db):
    db.insert_work_order(
        work_order_id="AO-003",
        work_order_type="corrective",
        registered_date=date(2026, 6, 1),
        completed_date=date(2026, 6, 2),
        fault_description="Feil",
        technical_description="Reparert",
        device_reg=18839,
        serial_number="8204570",
        import_filename="test.xlsx",
    )
    results = match_work_orders(db)
    assert len(results) == 1
    assert results[0]["event_type"] == "unrelated"


def test_match_saves_feedback_labels(db):
    db.insert_alert(
        device_serial="000000008204570",
        timestamp=datetime(2026, 1, 7, 6, 0),
        alert_level="critical",
        anomaly_score=-0.05,
        fleet_percentile=0.99,
    )
    db.insert_work_order(
        work_order_id="AO-001",
        work_order_type="corrective",
        registered_date=date(2026, 1, 9),
        completed_date=None,
        fault_description="Sensor defekt",
        technical_description="",
        device_reg=18839,
        serial_number="8204570",
        import_filename="test.xlsx",
    )

    match_work_orders(db)
    labels = db.get_feedback_labels()
    assert len(labels) == 1
    assert labels[0]["event_type"] == "confirmed_fault"
    assert labels[0]["source"] == "auto"
