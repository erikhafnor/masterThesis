"""Service configuration loader.

Defines a hierarchy of dataclass-based configuration sections that map
directly to top-level keys in the YAML config file.  Use `load_config`
to parse a YAML file into a `ServiceConfig` object.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ServerConfig:
    """HTTP server bind settings.

    Attributes:
        host: Interface address to bind the Uvicorn server on.
            Defaults to ``"0.0.0.0"`` (all interfaces).
        port: TCP port to listen on. Defaults to 8000.
    """

    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class QuestDBConfig:
    """Connection settings for the QuestDB telemetry store.

    Attributes:
        url: Full URL of the QuestDB HTTP query endpoint.
            Defaults to ``"http://localhost:9000/exec"``.
    """

    url: str = "http://localhost:9000/exec"


@dataclass
class SMTPConfig:
    """SMTP relay settings used for alert e-mail notifications.

    Attributes:
        host: Hostname of the SMTP relay server. Empty string disables
            e-mail delivery.
        port: SMTP submission port. Defaults to 587 (STARTTLS).
        from_addr: Sender address that appears in the ``From`` header.
            Mapped from the YAML key ``from`` (Python keyword conflict).
        recipients: List of e-mail addresses that receive alert
            notifications.
    """

    host: str = ""
    port: int = 587
    from_addr: str = ""
    recipients: list[str] = field(default_factory=list)


@dataclass
class SchedulerConfig:
    """APScheduler cron expressions and polling intervals.

    Attributes:
        inference_cron: Cron expression for the daily inference job.
            Defaults to ``"0 6 * * *"`` (06:00 every day).
        label_match_cron: Cron expression for the work-order label
            matching job. Defaults to ``"0 7 * * *"`` (07:00 every day).
        health_check_interval_minutes: How often the health-check job
            runs, in minutes. Defaults to 15.
        timezone: Timezone name used to interpret cron expressions.
            Defaults to ``"Europe/Oslo"``.
    """

    inference_cron: str = "0 6 * * *"
    label_match_cron: str = "0 7 * * *"
    health_check_interval_minutes: int = 15
    timezone: str = "Europe/Oslo"


@dataclass
class PathsConfig:
    """Filesystem paths used at runtime.

    Attributes:
        model: Path to the serialised anomaly-detection model file
            (joblib format). Defaults to the iForest artefact produced
            by the training pipeline.
        database: Path to the SQLite service database. Defaults to
            ``"data/pdm_service.db"``.
        medusa_import_dir: Directory where Medusa XLSX exports are
            staged for import. Defaults to ``"data/medusa_imports/"``.
        log_dir: Directory used for rotating log files. Defaults to
            ``"data/logs/"``.
    """

    model: Path = Path("outputs/models/iforest/iforest.joblib")
    database: Path = Path("data/pdm_service.db")
    medusa_import_dir: Path = Path("data/medusa_imports/")
    log_dir: Path = Path("data/logs/")


@dataclass
class AlertsConfig:
    """Thresholds and cooldown settings for alert generation.

    Attributes:
        threshold_percentile: Fleet-percentile value above which an
            anomaly score triggers at least a *warning* alert.
            Defaults to 0.95 (95th percentile).
        sustained_periods: Number of consecutive inference periods a
            device must stay above the threshold before a *critical*
            alert is raised. Defaults to 3.
        email_cooldown_hours: Minimum hours between successive e-mail
            notifications for the same device. Defaults to 24.
    """

    threshold_percentile: float = 0.95
    sustained_periods: int = 3
    email_cooldown_hours: int = 24


@dataclass
class ServiceConfig:
    """Top-level configuration container for the PdM service.

    Aggregates all configuration sections.  Loaded from a YAML file via
    `load_config`.

    Attributes:
        server: HTTP server bind settings.
        questdb: QuestDB telemetry store connection settings.
        smtp: SMTP relay settings for e-mail notifications.
        scheduler: APScheduler cron and interval settings.
        paths: Filesystem paths used at runtime.
        alerts: Alert threshold and cooldown settings.
    """

    server: ServerConfig = field(default_factory=ServerConfig)
    questdb: QuestDBConfig = field(default_factory=QuestDBConfig)
    smtp: SMTPConfig = field(default_factory=SMTPConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)


def load_config(path: Path) -> ServiceConfig:
    """Load and validate service configuration from a YAML file.

    Reads the YAML file at *path*, maps each top-level key to the
    corresponding configuration dataclass, and returns a fully populated
    `ServiceConfig`.  Missing sections fall back to their dataclass
    defaults, so a minimal YAML file only needs to override what differs
    from the defaults.

    Args:
        path: Filesystem path to the YAML configuration file.

    Returns:
        A `ServiceConfig` with all sub-sections populated from the YAML
        content or their dataclass defaults.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the YAML content is not a mapping (i.e. not a
            dict at the top level).
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(raw).__name__}")

    smtp_raw = raw.get("smtp", {})
    paths_raw = raw.get("paths", {})

    return ServiceConfig(
        server=ServerConfig(**raw.get("server", {})),
        questdb=QuestDBConfig(**raw.get("questdb", {})),
        smtp=SMTPConfig(
            host=smtp_raw.get("host", ""),
            port=smtp_raw.get("port", 587),
            from_addr=smtp_raw.get("from", ""),
            recipients=smtp_raw.get("recipients", []),
        ),
        scheduler=SchedulerConfig(**raw.get("scheduler", {})),
        paths=PathsConfig(
            model=Path(paths_raw.get("model", "outputs/models/iforest/iforest.joblib")),
            database=Path(paths_raw.get("database", "data/pdm_service.db")),
            medusa_import_dir=Path(paths_raw.get("medusa_import_dir", "data/medusa_imports/")),
            log_dir=Path(paths_raw.get("log_dir", "data/logs/")),
        ),
        alerts=AlertsConfig(**raw.get("alerts", {})),
    )
