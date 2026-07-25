"""QThread wrapper around app.core.benchmark.run_benchmark - kept separate
from OCRWorker since it runs against a single page/document and reports
per-variant results incrementally rather than per-page batch progress."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from app.core.benchmark import build_curated_variants, run_benchmark
from app.core.control import JobControl
from app.core.model_manager import ModelManager
from app.core.models import BenchmarkRunResult, OCRConfig


class BenchmarkWorker(QThread):
    variant_finished = Signal(object)  # BenchmarkRunResult
    benchmark_finished = Signal(list)  # list[BenchmarkRunResult]
    benchmark_failed = Signal(str)

    def __init__(self, model_manager: ModelManager, parent=None) -> None:
        super().__init__(parent)
        self._model_manager = model_manager
        self._control = JobControl()
        self._pdf_path: Path | None = None
        self._page_index: int = 0
        self._ocr_config: OCRConfig | None = None

    def start_benchmark(self, pdf_path: Path, page_index: int, ocr_config: OCRConfig) -> None:
        self._pdf_path = pdf_path
        self._page_index = page_index
        self._ocr_config = ocr_config
        self.start()

    def cancel(self) -> None:
        self._control.cancel()

    def run(self) -> None:
        if self._pdf_path is None or self._ocr_config is None:
            return
        try:
            self._control.reset()
            results: list[BenchmarkRunResult] = run_benchmark(
                self._pdf_path,
                self._page_index,
                self._model_manager,
                self._ocr_config,
                variants=build_curated_variants(),
                control=self._control,
                on_variant_done=self.variant_finished.emit,
            )
            self.benchmark_finished.emit(results)
        except Exception as exc:  # noqa: BLE001
            self.benchmark_failed.emit(str(exc))
