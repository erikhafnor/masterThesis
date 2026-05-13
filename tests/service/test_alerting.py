import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from ventilator_pdm.service.alerting import compose_alert_email, compose_error_email
from ventilator_pdm.service.config import SMTPConfig


def test_compose_alert_email_single_device():
    alerts = [
        {
            "device_serial": "000000008204570",
            "device_reg": 18839,
            "alert_level": "critical",
            "fleet_percentile": 0.99,
            "timestamp": "2026-04-15T06:00:00",
            "top_features": "O2 flow sensor health (38%), FiO2 deviation (22%)",
        },
    ]
    subject, body = compose_alert_email(alerts, base_url="http://server:8000")

    assert "1 device" in subject
    assert "18839" in body
    assert "CRITICAL" in body
    assert "http://server:8000/device/000000008204570" in body


def test_compose_alert_email_multiple_devices():
    alerts = [
        {"device_serial": "s1", "device_reg": 18817, "alert_level": "warning",
         "fleet_percentile": 0.96, "timestamp": "2026-04-15T06:00:00", "top_features": ""},
        {"device_serial": "s2", "device_reg": 18839, "alert_level": "critical",
         "fleet_percentile": 0.99, "timestamp": "2026-04-15T06:00:00", "top_features": ""},
    ]
    subject, body = compose_alert_email(alerts, base_url="http://server:8000")
    assert "2 devices" in subject


def test_compose_error_email():
    subject, body = compose_error_email(
        job_name="daily_inference",
        error_message="QuestDB connection refused",
    )
    assert "Error" in subject
    assert "daily_inference" in body
    assert "QuestDB connection refused" in body
