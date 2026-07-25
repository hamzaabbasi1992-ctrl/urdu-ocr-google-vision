"""Main application window: wires the file picker, preprocessing/OCR
settings, page preview, text/confidence view, benchmark mode, and batch
processing together. This is the only widget-composition point - the
widgets themselves stay decoupled and only emit/receive signals."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.job import JobStatus, OCRJob
from app.core.model_manager import ModelManager
from app.core.paths import default_output_dir
from app.gui.theme import stylesheet_for
from app.gui.widgets.batch_panel import BatchPanel
from app.gui.widgets.benchmark_panel import BenchmarkPanel
from app.gui.widgets.ocr_settings_panel import OCRSettingsPanel
from app.gui.widgets.page_preview import PagePreviewWidget
from app.gui.widgets.pdf_picker import PdfPickerWidget
from app.gui.widgets.preprocess_panel import PreprocessPanel
from app.gui.widgets.text_view import TextViewWidget
from app.workers.benchmark_worker import BenchmarkWorker
from app.workers.ocr_worker import OCRWorker

_LOGGER = logging.getLogger("urdu_ocr.gui.main_window")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Urdu OCR")
        self.resize(1400, 900)

        self._model_manager = ModelManager()
        self._output_folder = default_output_dir()
        self._dark_mode = True
        self._ocr_worker: OCRWorker | None = None
        self._benchmark_worker: BenchmarkWorker | None = None

        self.pdf_picker = PdfPickerWidget()
        self.preprocess_panel = PreprocessPanel()
        self.ocr_settings_panel = OCRSettingsPanel()
        self.page_preview = PagePreviewWidget()
        self.text_view = TextViewWidget()
        self.benchmark_panel = BenchmarkPanel()
        self.batch_panel = BatchPanel()

        self.page_preview.bind_config_providers(
            self.preprocess_panel.get_config, self.ocr_settings_panel.get_config
        )

        self._build_menu()
        self._build_layout()
        self._wire_signals()
        self._apply_theme()

    # ---- layout ----------------------------------------------------

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        add_action = file_menu.addAction("Add PDF(s)...")
        add_action.triggered.connect(self.pdf_picker.add_files_dialog)
        output_action = file_menu.addAction("Set Output Folder...")
        output_action.triggered.connect(self._choose_output_folder)
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        view_menu = self.menuBar().addMenu("&View")
        self.dark_mode_action = view_menu.addAction("Dark Mode")
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.setChecked(True)
        self.dark_mode_action.toggled.connect(self._on_dark_mode_toggled)

    def _build_layout(self) -> None:
        left_tabs = QTabWidget()
        left_tabs.addTab(self.pdf_picker, "Files")
        left_tabs.addTab(self.preprocess_panel, "Preprocessing")
        left_tabs.addTab(self.ocr_settings_panel, "OCR Settings")

        center_tabs = QTabWidget()
        center_tabs.addTab(self.page_preview, "Preview")
        center_tabs.addTab(self.text_view, "Text && Confidence")
        center_tabs.addTab(self.benchmark_panel, "Benchmark")

        export_row = QHBoxLayout()
        self.export_txt_button = QPushButton("Export TXT")
        self.export_docx_button = QPushButton("Export DOCX")
        self.export_pdf_button = QPushButton("Export Searchable PDF")
        self.export_json_button = QPushButton("Export JSON")
        for b in (self.export_txt_button, self.export_docx_button, self.export_pdf_button, self.export_json_button):
            b.setEnabled(False)
            export_row.addWidget(b)

        center_container = QWidget()
        center_layout = QVBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(center_tabs)
        center_layout.addLayout(export_row)

        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(left_tabs)
        top_splitter.addWidget(center_container)
        top_splitter.setStretchFactor(1, 1)

        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(self.batch_panel)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)

        self.setCentralWidget(main_splitter)

    # ---- wiring ------------------------------------------------------

    def _wire_signals(self) -> None:
        self.pdf_picker.files_changed.connect(self._on_files_changed)
        self.pdf_picker.list_widget.currentRowChanged.connect(self._on_selected_file_changed)

        self.batch_panel.start_clicked.connect(self._on_start_batch)
        self.batch_panel.pause_clicked.connect(lambda: self._ocr_worker and self._ocr_worker.pause())
        self.batch_panel.resume_clicked.connect(lambda: self._ocr_worker and self._ocr_worker.resume())
        self.batch_panel.cancel_clicked.connect(lambda: self._ocr_worker and self._ocr_worker.cancel())

        self.benchmark_panel.run_clicked.connect(self._on_run_benchmark)
        self.benchmark_panel.apply_clicked.connect(self._on_apply_benchmark_result)

        self.export_txt_button.clicked.connect(lambda: self._export("txt"))
        self.export_docx_button.clicked.connect(lambda: self._export("docx"))
        self.export_pdf_button.clicked.connect(lambda: self._export("pdf"))
        self.export_json_button.clicked.connect(lambda: self._export("json"))

    def _on_dark_mode_toggled(self, checked: bool) -> None:
        self._dark_mode = checked
        self._apply_theme()

    def _apply_theme(self) -> None:
        from PySide6.QtWidgets import QApplication

        QApplication.instance().setStyleSheet(stylesheet_for(self._dark_mode))

    # ---- files ---------------------------------------------------------

    def _on_files_changed(self, files: list[Path]) -> None:
        jobs = [OCRJob(source_path=p) for p in files]
        self.batch_panel.set_jobs(jobs)
        self._jobs = jobs
        if files and self.pdf_picker.list_widget.currentRow() < 0:
            self.pdf_picker.list_widget.setCurrentRow(0)

    def _on_selected_file_changed(self, row: int) -> None:
        files = self.pdf_picker.files()
        if 0 <= row < len(files):
            self.page_preview.set_pdf(files[row])

    def _choose_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select output folder", str(self._output_folder))
        if folder:
            self._output_folder = Path(folder)

    # ---- batch processing -----------------------------------------------

    def _on_start_batch(self) -> None:
        jobs = getattr(self, "_jobs", [])
        if not jobs:
            QMessageBox.warning(self, "No files", "Add at least one PDF first.")
            return

        self._ocr_worker = OCRWorker(self._model_manager)
        self._ocr_worker.job_updated.connect(self.batch_panel.update_job)
        self._ocr_worker.job_updated.connect(self._on_job_updated)
        self._ocr_worker.page_processed.connect(self._on_page_processed)
        self._ocr_worker.stage_changed.connect(lambda job, msg, _indet: self.batch_panel.set_stage(msg))
        self._ocr_worker.batch_finished.connect(self._on_batch_finished)
        self._ocr_worker.batch_failed.connect(self._on_batch_failed)

        self.batch_panel.set_running(True)
        self._ocr_worker.start_batch(
            jobs,
            self.preprocess_panel.get_config(),
            self.ocr_settings_panel.get_config(),
            self._output_folder,
        )

    def _on_job_updated(self, job: OCRJob) -> None:
        if job.status == JobStatus.DONE and job.result is not None:
            self._last_result = job.result
            for b in (self.export_txt_button, self.export_docx_button, self.export_pdf_button, self.export_json_button):
                b.setEnabled(True)

    def _on_page_processed(self, job: OCRJob, page) -> None:
        self.text_view.set_page(page)

    def _on_batch_finished(self) -> None:
        self.batch_panel.set_running(False)
        self.batch_panel.set_stage("Batch finished.")

    def _on_batch_failed(self, message: str) -> None:
        self.batch_panel.set_running(False)
        QMessageBox.critical(self, "Batch failed", message)

    # ---- benchmark -------------------------------------------------------

    def _on_run_benchmark(self) -> None:
        files = self.pdf_picker.files()
        row = self.pdf_picker.list_widget.currentRow()
        if not (0 <= row < len(files)):
            QMessageBox.warning(self, "No file selected", "Select a PDF in the Files tab first.")
            return

        self.benchmark_panel.clear()
        pdf_path = files[row]
        page_index = self.page_preview.current_page_index()

        self._benchmark_worker = BenchmarkWorker(self._model_manager)
        self._benchmark_worker.variant_finished.connect(self.benchmark_panel.add_result)
        self._benchmark_worker.benchmark_finished.connect(lambda _results: self.benchmark_panel.highlight_best())
        self._benchmark_worker.benchmark_failed.connect(
            lambda msg: QMessageBox.critical(self, "Benchmark failed", msg)
        )
        self._benchmark_worker.start_benchmark(pdf_path, page_index, self.ocr_settings_panel.get_config())

    def _on_apply_benchmark_result(self, result) -> None:
        self.preprocess_panel.load_config(result.preprocess)
        QMessageBox.information(
            self, "Pipeline applied", f"Applied preprocessing settings from '{result.variant_name}'."
        )

    # ---- export ------------------------------------------------------

    def _export(self, kind: str) -> None:
        result = getattr(self, "_last_result", None)
        if result is None:
            QMessageBox.warning(self, "Nothing to export", "Run a batch first.")
            return

        extensions = {"txt": ("*.txt", "output.txt"), "docx": ("*.docx", "output.docx"),
                      "pdf": ("*.pdf", "output_searchable.pdf"), "json": ("*.json", "output.json")}
        pattern, default_name = extensions[kind]
        path_str, _ = QFileDialog.getSaveFileName(self, "Export", str(self._output_folder / default_name), pattern)
        if not path_str:
            return

        output_path = Path(path_str)
        if kind == "txt":
            from app.core.exporters.txt_exporter import export_txt
            export_txt(result, output_path)
        elif kind == "docx":
            from app.core.exporters.docx_exporter import export_docx
            export_docx(result, output_path)
        elif kind == "pdf":
            from app.core.exporters.searchable_pdf_exporter import export_searchable_pdf
            export_searchable_pdf(result, output_path)
        elif kind == "json":
            from app.core.exporters.json_exporter import export_json
            export_json(result, output_path)
