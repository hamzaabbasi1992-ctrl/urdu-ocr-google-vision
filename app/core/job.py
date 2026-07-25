"""A single PDF's place in a batch: its status, timing, and eventual result."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from app.core.models import DocumentResult


class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class OCRJob:
    source_path: Path
    status: JobStatus = JobStatus.QUEUED
    total_pages: int | None = None
    processed_pages: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    error_message: str | None = None
    output_folder: Path | None = None
    result: DocumentResult | None = None

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started_at

    @property
    def progress_fraction(self) -> float:
        if not self.total_pages:
            return 0.0
        return min(1.0, self.processed_pages / self.total_pages)

    @property
    def estimated_remaining_seconds(self) -> float | None:
        """Extrapolate remaining wall-clock time from progress so far."""
        if self.status != JobStatus.RUNNING or self.processed_pages <= 0 or not self.total_pages:
            return None
        rate = self.elapsed_seconds / self.processed_pages  # wall-clock seconds per page
        remaining_pages = max(0, self.total_pages - self.processed_pages)
        return rate * remaining_pages
