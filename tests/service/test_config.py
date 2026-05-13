import pytest
from pathlib import Path
from ventilator_pdm.service.config import ServiceConfig, load_config


def test_load_config_from_yaml(tmp_path):
    yaml_content = """
server:
  host: "127.0.0.1"
  port: 9000

questdb:
  url: "http://localhost:9000/exec"

smtp:
  host: "mail.test.no"
  port: 587
  from: "test@test.no"
  recipients: ["a@test.no"]

scheduler:
  inference_cron: "0 6 * * *"
  label_match_cron: "0 7 * * *"
  health_check_interval_minutes: 15
  timezone: "Europe/Oslo"

paths:
  model: "outputs/models/iforest/iforest.joblib"
  database: "data/pdm_service.db"
  medusa_import_dir: "data/medusa_imports/"
  log_dir: "data/logs/"

alerts:
  threshold_percentile: 0.95
  sustained_periods: 3
  email_cooldown_hours: 24
"""
    config_file = tmp_path / "service.yaml"
    config_file.write_text(yaml_content)

    cfg = load_config(config_file)

    assert isinstance(cfg, ServiceConfig)
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.port == 9000
    assert cfg.questdb.url == "http://localhost:9000/exec"
    assert cfg.smtp.host == "mail.test.no"
    assert cfg.smtp.from_addr == "test@test.no"
    assert cfg.smtp.recipients == ["a@test.no"]
    assert cfg.scheduler.timezone == "Europe/Oslo"
    assert cfg.paths.model == Path("outputs/models/iforest/iforest.joblib")
    assert cfg.alerts.threshold_percentile == 0.95
    assert cfg.alerts.sustained_periods == 3
    assert cfg.alerts.email_cooldown_hours == 24


def test_load_config_empty_file(tmp_path):
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")

    cfg = load_config(config_file)

    assert isinstance(cfg, ServiceConfig)
    assert cfg.server.host == "0.0.0.0"
    assert cfg.server.port == 8000


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))
