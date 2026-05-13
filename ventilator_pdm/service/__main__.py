"""Entry point for ``python -m pdm.service``.

Parses CLI arguments, configures rotating file logging, initialises the
database, builds the FastAPI application, and starts a uvicorn server.

CLI usage::

    python -m pdm.service [--config PATH]

Options:
    --config PATH   Path to the ``service.yaml`` configuration file.
                    Defaults to ``configs/service.yaml``.
"""
import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import uvicorn

from ventilator_pdm.service.config import load_config
from ventilator_pdm.service.database import Database
from ventilator_pdm.service.app import create_app


def setup_logging(log_dir: Path) -> None:
    """Configure rotating file logging for the service process.

    Creates ``log_dir`` if it does not exist, attaches a
    :class:`~logging.handlers.RotatingFileHandler` writing to
    ``service.log`` (max 10 MB, 5 backups), and sets the root logger level
    to ``INFO``.

    Args:
        log_dir: Directory in which ``service.log`` will be created.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "service.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    ))
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)


def main() -> None:
    """Start the PdM service.

    Parses the ``--config`` CLI argument, loads the service configuration,
    sets up file logging, initialises the SQLite database, creates the
    FastAPI application, and runs it with uvicorn on the host and port
    specified in the configuration.
    """
    parser = argparse.ArgumentParser(description="PdM Service")
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/service.yaml"),
        help="Path to service.yaml config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.paths.log_dir)

    db = Database(config.paths.database)
    app = create_app(config, db)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
