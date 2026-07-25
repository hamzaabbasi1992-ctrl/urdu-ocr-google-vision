"""Shared contract for output writers, so a future format is a drop-in addition."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.core.models import DocumentResult


class Exporter(Protocol):
    def __call__(self, result: DocumentResult, output_path: Path) -> None: ...
