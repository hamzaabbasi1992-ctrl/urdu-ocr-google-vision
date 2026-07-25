"""Application-wide logging: a rotating app log plus one log file per batch run.

The app log captures everything for troubleshooting across sessions. The
batch log is scoped to a single run and lists per-document/page timing and
any errors/warnings for that run - it is written next to that run's output.
"""

from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.paths import logs_dir

APP_LOGGER_NAME = "urdu_ocr"


def setup_app_logging() -> logging.Logger:
    logger = logging.getLogger(APP_LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = RotatingFileHandler(
        logs_dir() / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger


def create_batch_log(output_root: Path) -> tuple[logging.Logger, Path]:
    """Create a fresh logger writing to a timestamped log file for one batch run."""
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_root / f"batch_{timestamp}.log"

    logger = logging.getLogger(f"{APP_LOGGER_NAME}.batch.{timestamp}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)

    return logger, log_path
