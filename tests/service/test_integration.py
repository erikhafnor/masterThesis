# tests/service/test_integration.py
"""End-to-end integration test for PdM service."""
import pytest
from datetime import datetime, date
from pathlib import Path
from openpyxl import Workbook
from fastapi.testclient import TestClient

from ventilator_pdm.service.config import ServiceConfig
from ventilator_pdm.service.database import Database
from ventilator_pdm.service.app import create_app


@pytest.fixture
def setup(tmp_path):
    cfg = ServiceConfig()
    cfg.paths.database = tmp_path / "test.db"
    cfg.paths.medusa_import_dir = tmp_path / "imports"
    cfg.paths.medusa_import_dir.mkdir()

    db = Database(cfg.paths.database)
    db.initialize()
    db.insert_model_version("outputs/models/iforest/iforest.joblib", 0, 0)

    app = create_app(cfg, db, start_scheduler=False)
    client = TestClient(app)
    return client, db, cfg, tmp_path


def test_full_workflow(setup):
    client, db, cfg, tmp_path = setup

    # 1. All pages load
    for path in ["/", "/fleet", "/alerts", "/import", "/learning", "/reports"]:
        resp = client.get(path)
        assert resp.status_code == 200, f"Page {path} failed: {resp.status_code}"

    # 2. Insert some alerts
    db.insert_alert("000000008204570", datetime(2026, 4, 1), "warning", -0.1, 0.97)
    db.insert_alert("000000008204570", datetime(2026, 4, 2), "critical", -0.05, 0.99)

    # 3. Device page loads with alerts
    resp = client.get("/device/000000008204570")
    assert resp.status_code == 200

    # 4. Fleet page shows the device
    resp = client.get("/fleet")
    assert resp.status_code == 200

    # 5. Export works
    resp = client.get("/api/reports/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
