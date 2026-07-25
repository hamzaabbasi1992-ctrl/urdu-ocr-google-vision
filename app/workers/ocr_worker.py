"""QThread that drives the JobQueue and re-publishes its progress as Qt
signals.

This is the only place the core pipeline and Qt meet: everything in
app.core is plain Python with no Qt dependency, and everything in app.gui
only reacts to signals from here.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from app.core.job import OCRJob
from app.core.job_queue import JobQueue
from app.core.logging_setup import create_batch_log
from app.core.model_manager import ModelManager
from app.core.models import OCRConfig, PageResult, PreprocessConfig
from app.core.control import JobControl


class OCRWorker(QThread):
    job_updated = Signal(object)  # OCRJob
    page_processed = Signal(object, object)  # OCRJob, PageResult
    stage_changed = Signal(object, str, bool)  # OCRJob, message, indeterminate
    batch_finished = Signal()
    batch_failed = Signal(str)

    def __init__(self, model_manager: ModelManager, parent=None) -> None:
        super().__init__(parent)
        self._queue = JobQueue(model_manager)
        self._control = JobControl()
        self._jobs: list[OCRJob] = []
        self._preprocess_config: PreprocessConfig | None = None
        self._ocr_config: OCRConfig | None = None
        self._output_root: Path | None = None

    def start_batch(
        self,
        jobs: list[OCRJob],
        preprocess_config: PreprocessConfig,
        ocr_config: OCRConfig,
        output_root: Path,
    ) -> None:
        self._jobs = jobs
        self._preprocess_config = preprocess_config
        self._ocr_config = ocr_config
        self._output_root = output_root
        self.start()

    def pause(self) -> None:
        self._control.pause()

    def resume(self) -> None:
        self._control.resume()

    def cancel(self) -> None:
        self._control.cancel()

    def is_paused(self) -> bool:
        return self._control.is_paused()

    def run(self) -> None:
        if self._preprocess_config is None or self._ocr_config is None or self._output_root is None:
            return

        batch_logger, log_path = create_batch_log(self._output_root)
        batch_logger.info("Batch started: %d document(s)", len(self._jobs))
        try:
            self._queue.run(
                self._jobs,
                self._preprocess_config,
                self._ocr_config,
                self._output_root,
                self._control,
                batch_logger,
                on_job_update=self.job_updated.emit,
                on_page=self.page_processed.emit,
                on_stage=self.stage_changed.emit,
            )
            batch_logger.info("Batch finished. Log written to %s", log_path)
            self.batch_finished.emit()
        except Exception as exc:  # noqa: BLE001 - surface unexpected failures to the GUI, don't crash silently
            batch_logger.error("Batch failed unexpectedly: %s", exc, exc_info=True)
            self.batch_failed.emit(str(exc))
