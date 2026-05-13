# tests/service/test_app.py
import pytest
from pathlib import Path
from unittest.mock import patch
from ventilator_pdm.service.config import ServiceConfig


@pytest.fixture
def config(tmp_path):
    cfg = ServiceConfig()
    cfg.paths.database = tmp_path / "test.db"
    cfg.paths.model = Path("outputs/models/iforest/iforest.joblib")
    return cfg


@pytest.fixture
def client(config):
    from ventilator_pdm.service.app import create_app
    from ventilator_pdm.service.database import Database

    db = Database(config.paths.database)
    db.initialize()

    app = create_app(config, db, start_scheduler=False)
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_health_page_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "System Health" in resp.text


def test_health_page_shows_pipeline_status(client):
    resp = client.get("/")
    assert resp.status_code == 200
    # Should show status even with no data yet
    assert "No inference runs yet" in resp.text or "Pipeline" in resp.text


def test_fleet_page_returns_200(client):
    resp = client.get("/fleet")
    assert resp.status_code == 200
    assert "Fleet Overview" in resp.text


def test_device_page_returns_200(client):
    resp = client.get("/device/000000008204570")
    assert resp.status_code == 200
    assert "Device" in resp.text


def test_device_page_unknown_serial_returns_404(client):
    resp = client.get("/device/nonexistent")
    assert resp.status_code == 404


def test_alerts_page_returns_200(client):
    resp = client.get("/alerts")
    assert resp.status_code == 200
    assert "Alert History" in resp.text


def test_import_page_returns_200(client):
    resp = client.get("/import")
    assert resp.status_code == 200
    assert "Medusa" in resp.text


def test_learning_page_returns_200(client):
    resp = client.get("/learning")
    assert resp.status_code == 200
    assert "Active Learning" in resp.text


def test_reports_page_returns_200(client):
    resp = client.get("/reports")
    assert resp.status_code == 200
    assert "Reports" in resp.text


def test_export_api_returns_zip(client):
    resp = client.get("/api/reports/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
