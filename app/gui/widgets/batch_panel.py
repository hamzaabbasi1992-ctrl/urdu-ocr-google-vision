"""Batch queue: job list with status, overall progress bar, and
start/pause/resume/cancel controls."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.job import JobStatus, OCRJob

_STATUS_LABELS = {
    JobStatus.QUEUED: "Queued",
    JobStatus.RUNNING: "Running",
    JobStatus.DONE: "Done",
    JobStatus.ERROR: "Error",
    JobStatus.CANCELLED: "Cancelled",
}


class BatchPanel(QWidget):
    start_clicked = Signal()
    pause_clicked = Signal()
    resume_clicked = Signal()
    cancel_clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._items: dict[str, QListWidgetItem] = {}
        self._paused = False

        self.list_widget = QListWidget()
        self.progress_bar = QProgressBar()
        self.stage_label = QLabel("")

        self.start_button = QPushButton("Start Batch")
        self.pause_button = QPushButton("Pause")
        self.cancel_button = QPushButton("Cancel")
        self.pause_button.setEnabled(False)
        self.cancel_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_clicked.emit)
        self.pause_button.clicked.connect(self._on_pause_clicked)
        self.cancel_button.clicked.connect(self.cancel_clicked.emit)

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.pause_button)
        button_row.addWidget(self.cancel_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.list_widget)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.stage_label)
        layout.addLayout(button_row)

    def _on_pause_clicked(self) -> None:
        if self._paused:
            self._paused = False
            self.pause_button.setText("Pause")
            self.resume_clicked.emit()
        else:
            self._paused = True
            self.pause_button.setText("Resume")
            self.pause_clicked.emit()

    def set_jobs(self, jobs: list[OCRJob]) -> None:
        self.list_widget.clear()
        self._items.clear()
        for job in jobs:
            item = QListWidgetItem(self._label_for(job))
            self.list_widget.addItem(item)
            self._items[str(job.source_path)] = item

    def update_job(self, job: OCRJob) -> None:
        item = self._items.get(str(job.source_path))
        if item is not None:
            item.setText(self._label_for(job))

        done = sum(1 for j in [job] if j.status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED))
        if job.total_pages:
            self.progress_bar.setMaximum(job.total_pages)
            self.progress_bar.setValue(job.processed_pages)

    def set_stage(self, message: str) -> None:
        self.stage_label.setText(message)

    def set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.cancel_button.setEnabled(running)
        if not running:
            self._paused = False
            self.pause_button.setText("Pause")

    @staticmethod
    def _label_for(job: OCRJob) -> str:
        status = _STATUS_LABELS[job.status]
        progress = f"{job.processed_pages}/{job.total_pages}" if job.total_pages else "-"
        return f"{job.source_path.name}  [{status}]  {progress}"
