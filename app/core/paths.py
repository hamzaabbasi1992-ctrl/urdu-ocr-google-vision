"""Central definitions of where the app stores its persistent, non-project data.

Kept as a single small module so config, logging, and the model cache all
agree on the same locations without importing each other.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "UrduOCR"


def app_data_dir() -> Path:
    """Roaming config location: %APPDATA%\\UrduOCR."""
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    path = Path(base) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def local_app_data_dir() -> Path:
    """Local (non-roaming) cache location: %LOCALAPPDATA%\\UrduOCR."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    path = Path(base) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_file() -> Path:
    return app_data_dir() / "config.json"


def logs_dir() -> Path:
    path = app_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_cache_dir() -> Path:
    path = local_app_data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def temp_dir() -> Path:
    """Per-run scratch space for intermediate page renders. Callers clean up
    their own subfolders once a page/document's final result is committed."""
    path = local_app_data_dir() / "temp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_dir() -> Path:
    return Path.home() / "Documents" / "Urdu OCR Output"
