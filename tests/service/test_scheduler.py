# tests/service/test_scheduler.py
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from pathlib import Path
from ventilator_pdm.service.database import Database
from ventilator_pdm.service.config import ServiceConfig, load_config
from ventilator_pdm.service.scheduler import run_health_check


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    return database


@pytest.fixture
def config(tmp_path):
    """Minimal config for testing."""
    cfg = ServiceConfig()
    cfg.paths.database = tmp_path / "test.db"
    cfg.questdb.url = "http://localhost:9000/exec"
    return cfg


def test_health_check_logs_success_when_questdb_reachable(db, config):
    with patch("pdm.service.scheduler.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        run_health_check(db, config)

    entries = db.get_health_log(job_name="health_check", limit=1)
    assert len(entries) == 1
    assert entries[0]["status"] == "success"


def test_health_check_logs_failure_when_questdb_unreachable(db, config):
    with patch("pdm.service.scheduler.requests.get") as mock_get:
        mock_get.side_effect = ConnectionError("refused")
        run_health_check(db, config)

    entries = db.get_health_log(job_name="health_check", limit=1)
    assert len(entries) == 1
    assert entries[0]["status"] == "failure"
    assert "refused" in entries[0]["error_message"]
