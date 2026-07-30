"""Minimal PySide6 GUI for running Google Vision OCR against a single PDF, a
whole folder of PDFs, or several individually-picked PDF files (three input
modes, one radio group). All three feed the same OCRWorker queue and get
the same per-file behavior below (checkpointing, skip-if-already-done,
output named after the input file's own stem).

Deliberately small and single-file - this is NOT the old elaborate
multi-widget GUI (app/gui/), which is tied to the superseded 2-engine
architecture and was never wired to the new minimal pipeline. This wraps
only what's actually built and measured (PROJECT_SPEC.md Section 6):
PDFLoader -> PageRasterizer -> GoogleVisionEngine -> TextExporter.

No deskew/denoise/CLAHE here on purpose - measured to make Google Vision's
accuracy worse, not better (Section 6, GoogleVisionEngine entry). Google's
own document-OCR pipeline is already tuned; our extra smoothing loses fine
detail like diacritic dots.

Output is TXT plus an optional DOCX (checkbox, on by default) and an
optional searchable PDF (checkbox, off by default - see SearchablePDFExporter,
app/core/export/searchable_pdf_exporter.py). JSON export was not ported to
the new architecture and remains out of scope (explicit choice, not an
oversight).

GoogleVisionEngine is a cloud engine (PROJECT_SPEC.md Section 2, explicit
user sign-off) - the window title and a persistent red label say so, so
this is never mistaken for an offline tool.

Usage counting is a local estimate only (this app's own count of pages it
has sent this calendar month), not synced with Google Cloud's actual
billing - stated as such in the UI so it isn't mistaken for authoritative
billing data.

Pause/Resume: the worker checks a cooperative stop flag between pages (not
mid-API-call - a single Vision call is a couple of seconds, not worth the
complexity of interrupting one in flight). Whatever pages were already
recognized for the file in progress are still written out - both on a
deliberate pause and on an unexpected error - via try/finally, so a network
hiccup on page 80 of 100 doesn't discard pages 1-79. The "Resume" label
(swapped in for "Run OCR" after a pause/failure) and the underlying
checkpoint/resume mechanism (see below) are the same mechanism a plain
app close/crash recovers through too - Pause is not a special case, just a
deliberate, immediate version of the same recovery every interruption gets.

Two-page-spread splitting (opt-in checkbox): diagnosed from a real user
report of "so many mistakes" that turned out to be word-salad, not
character errors - the source PDF had two physical book pages scanned side
by side per PDF page, and reading order was grouping same-height lines
from both pages into one scrambled line. Splitting each page in half before
recognition (right half first, correct order for RTL Urdu) fixes this.

Blank pages (near-uniform white, e.g. chapter-break pages in a scanned
book) are detected before spending an API call on them - zero accuracy
cost, since there is genuinely no text to recognize, and it saves quota.

Transient per-page API failures get a couple of automatic retries with a
short pause before the whole run gives up on that file (which, per the
above, still keeps everything recognized up to that point).

Checkpointing: for large files, waiting until the very end to write anything
means a hard crash/kill (not a clean stop - the try/finally above already
covers that) loses the whole file's progress. Every _CHECKPOINT_INTERVAL_PAGES
pages, the accumulated text so far is written to the same output path,
overwriting the previous checkpoint - so a crash at page 733 loses at most
the last 49 pages, not all 733.

The file/folder picker dialogs (all three input modes) reopen in whatever
directory was last browsed (persisted via QSettings as "last_input_dir"),
so picking several files from the same place across multiple runs doesn't
mean re-navigating from scratch each time.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import date
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.export.docx_exporter import DocxExporter
from app.core.export.searchable_pdf_exporter import SearchablePDFExporter
from app.core.export.text_exporter import TextExporter, assemble_text_with_headings
from app.core.ingestion.pdf_loader import PDFLoader
from app.core.ingestion.page_rasterizer import PageRasterizer
from app.core.recognition.google_vision_engine import GoogleVisionEngine
from app.core.recognition.recognized_word import RecognizedWord
from app.core.usage_monitor import get_vision_api_request_count_this_month

_DPI_OPTIONS = [
    ("Fast (150 DPI)", 150),
    ("Balanced (200 DPI) - recommended", 200),
    ("High detail (300 DPI)", 300),
    ("Maximum (400 DPI)", 400),
]
_FREE_TIER_PAGES_PER_MONTH = 1000
_PRICE_PER_1000_PAGES_USD = 1.50
_RECOGNIZE_RETRIES = 2
_RECOGNIZE_RETRY_DELAY_SECONDS = 2.0
_BLANK_WHITE_MEAN_THRESHOLD = 250.0
_BLANK_STD_THRESHOLD = 3.0
_CHECKPOINT_INTERVAL_PAGES = 50


def _is_blank_page(image: np.ndarray) -> bool:
    """A page that's almost uniformly white has nothing to recognize -
    skipping it costs nothing accuracy-wise and saves an API call."""
    return float(image.mean()) > _BLANK_WHITE_MEAN_THRESHOLD and float(image.std()) < _BLANK_STD_THRESHOLD


def _checkpoint_path(output_path: Path) -> Path:
    return Path(str(output_path) + ".progress.json")


def _load_checkpoint(output_path: Path, dpi: int, split_spread: bool) -> tuple[int, str] | None:
    """Returns (last_completed_index, existing_text) - the 0-based PDF page
    index of the last page already written to output_path, and that text -
    if a resumable checkpoint exists for this exact output path and matches
    the current DPI/split_spread settings. Settings must match because
    resuming with different settings would produce an inconsistent file
    (half one DPI, half another)."""
    checkpoint = _checkpoint_path(output_path)
    if not checkpoint.exists() or not output_path.exists():
        return None
    try:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if state.get("dpi") != dpi or state.get("split_spread") != split_spread:
        return None
    last_index = state.get("last_completed_index")
    if not isinstance(last_index, int) or last_index < 0:
        return None
    try:
        # utf-8-sig strips the BOM TextExporter writes - reading plain utf-8
        # here would leave it embedded mid-file once this text is
        # re-exported with a fresh BOM of its own.
        existing_text = output_path.read_text(encoding="utf-8-sig")
    except OSError:
        return None
    return last_index, existing_text


def _save_checkpoint(output_path: Path, last_completed_index: int, dpi: int, split_spread: bool) -> None:
    state = {"last_completed_index": last_completed_index, "dpi": dpi, "split_spread": split_spread}
    _checkpoint_path(output_path).write_text(json.dumps(state), encoding="utf-8")


def _clear_checkpoint(output_path: Path) -> None:
    _checkpoint_path(output_path).unlink(missing_ok=True)


def _already_fully_done(output_path: Path) -> bool:
    """True if output_path holds a complete conversion from a previous run -
    the file exists and there's no dangling checkpoint sidecar (a sidecar
    only survives an interrupted run; _process_one_file deletes it on a
    clean finish). Used to skip whole files in a folder batch that already
    finished before a crash/stop hit a later file, instead of redoing them.

    Assumes the existing output really does cover what's being requested
    now (same file, same or wider page range) - if you deliberately want to
    redo a completed file with different settings or a different range,
    delete its output first or use a different output name."""
    return output_path.exists() and not _checkpoint_path(output_path).exists()


def _recognize_with_retry(engine: GoogleVisionEngine, image: np.ndarray):
    last_exc: Exception | None = None
    for attempt in range(_RECOGNIZE_RETRIES + 1):
        try:
            return engine.recognize(image)
        except Exception as exc:  # noqa: BLE001 - retry any transient failure, then give up and re-raise
            last_exc = exc
            if attempt < _RECOGNIZE_RETRIES:
                time.sleep(_RECOGNIZE_RETRY_DELAY_SECONDS)
    assert last_exc is not None
    raise last_exc


class OCRWorker(QThread):
    file_started = Signal(str, int)  # pdf name, page count in THIS run
    page_done = Signal(int, int)  # current page within this run (1-based), total in this run
    page_recognized = Signal(int, int, str)  # current page, total, text (or a blank-skip note)
    api_call_made = Signal()  # one real Vision API call happened - the only correct usage-counting signal
    file_done = Signal(str, str)  # pdf name, output path
    file_skipped = Signal(str)  # pdf name - already fully converted in a previous run
    all_done = Signal()
    stopped = Signal()
    failed = Signal(str)

    def __init__(
        self,
        pdf_paths: list[Path],
        output_paths: list[Path],
        credentials_path: str,
        dpi: int,
        page_ranges: list[tuple[int, int] | None],
        write_docx: bool = False,
        write_searchable_pdf: bool = False,
        split_spread: bool = False,
    ) -> None:
        """page_ranges: one (start_index, end_index) 0-based inclusive pair
        per pdf_path, or None to process every page of that PDF.

        split_spread: the source PDF has two physical book pages scanned
        side by side per PDF page (a common artifact of photographing an
        open book rather than scanning single sheets). Grouping words by
        raw vertical position - correct for a normal single-column page -
        merges same-height lines from the right page and left page into
        one scrambled "line" otherwise. When enabled, each page is split
        down the middle and each half is recognized as its own image, read
        right half first (correct order for RTL Urdu, confirmed by visible
        page numbers on the sample that motivated this)."""
        super().__init__()
        self._pdf_paths = pdf_paths
        self._output_paths = output_paths
        self._credentials_path = credentials_path
        self._dpi = dpi
        self._page_ranges = page_ranges
        self._write_docx = write_docx
        self._write_searchable_pdf = write_searchable_pdf
        self._split_spread = split_spread
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            engine = GoogleVisionEngine(credentials_path=self._credentials_path)
            rasterizer = PageRasterizer()

            for pdf_path, output_path, page_range in zip(self._pdf_paths, self._output_paths, self._page_ranges):
                if _already_fully_done(output_path):
                    self.file_skipped.emit(pdf_path.name)
                    continue
                stopped_here = self._process_one_file(pdf_path, output_path, page_range, engine, rasterizer)
                if stopped_here:
                    self.stopped.emit()
                    return

            self.all_done.emit()
        except Exception as exc:  # noqa: BLE001 - surface to the GUI, don't crash silently
            self.failed.emit(str(exc))

    def _process_one_file(self, pdf_path, output_path, page_range, engine, rasterizer) -> bool:
        """Returns True if a user-requested stop interrupted this file.
        Whatever pages were recognized before a stop or an exception are
        still written out via the finally block - only truly zero pages
        processed for this file results in no output file.

        Resumes automatically from the last checkpoint if one exists for
        this exact output_path and matches the current dpi/split_spread -
        so re-running the same job after a crash/kill doesn't reprocess
        (and re-bill) pages already done. See _load_checkpoint."""
        page_texts: list[str] = []
        resume_prefix = ""
        last_completed_index: int | None = None
        stopped_here = False
        completed_normally = False
        pdf_exporter = (
            SearchablePDFExporter(existing_path=output_path.with_suffix(".pdf"))
            if self._write_searchable_pdf
            else None
        )
        try:
            with PDFLoader(pdf_path) as loader:
                if page_range is None:
                    start, end = 0, loader.page_count - 1
                else:
                    start, end = page_range
                    end = min(end, loader.page_count - 1)

                resume = _load_checkpoint(output_path, self._dpi, self._split_spread)
                if resume is not None:
                    resumed_index, existing_text = resume
                    last_completed_index = resumed_index
                    if resumed_index + 1 > start:
                        resume_prefix = existing_text
                        start = resumed_index + 1
                        self.page_recognized.emit(
                            0, 0, f"Resuming from checkpoint - continuing after page {resumed_index + 1}"
                        )

                if start > end:
                    # Everything in the requested range is already covered
                    # by a previous checkpoint - nothing new to do.
                    return stopped_here

                total_in_run = max(0, end - start + 1)
                self.file_started.emit(pdf_path.name, total_in_run)

                for offset, index in enumerate(range(start, end + 1)):
                    if self._stop_event.is_set():
                        stopped_here = True
                        break
                    page_number = offset + 1
                    self.page_done.emit(page_number, total_in_run)
                    page = loader.get_page(index)
                    image = rasterizer.rasterize(page, dpi=self._dpi).image

                    text, words = self._recognize_page_image(engine, image)
                    if text.strip():
                        page_texts.append(text)
                    if pdf_exporter is not None:
                        pdf_exporter.add_page(image, words, dpi=self._dpi)
                    self.page_recognized.emit(page_number, total_in_run, text or "(blank page - skipped)")
                    last_completed_index = index

                    if page_texts and page_number % _CHECKPOINT_INTERVAL_PAGES == 0:
                        self._export_current(page_texts, output_path, resume_prefix, pdf_exporter)
                        _save_checkpoint(output_path, last_completed_index, self._dpi, self._split_spread)
                else:
                    # for-else: only runs if the loop finished without break -
                    # i.e. a real stop or exception leaves this False, so the
                    # checkpoint below is preserved (not cleared) in both cases.
                    completed_normally = True
            return stopped_here
        finally:
            if page_texts or resume_prefix:
                self._export_current(page_texts, output_path, resume_prefix, pdf_exporter)
                if completed_normally:
                    _clear_checkpoint(output_path)
                elif last_completed_index is not None:
                    _save_checkpoint(output_path, last_completed_index, self._dpi, self._split_spread)
                self.file_done.emit(pdf_path.name, str(output_path))
            if pdf_exporter is not None:
                pdf_exporter.close()

    def _export_current(
        self,
        page_texts: list[str],
        output_path: Path,
        resume_prefix: str = "",
        pdf_exporter: SearchablePDFExporter | None = None,
    ) -> None:
        new_text = "\n\n".join(page_texts)
        text = f"{resume_prefix}\n\n{new_text}" if resume_prefix and new_text else (resume_prefix or new_text)
        TextExporter().export(text, output_path)
        if self._write_docx:
            DocxExporter().export(text, output_path.with_suffix(".docx"))
        if pdf_exporter is not None:
            pdf_exporter.save(output_path.with_suffix(".pdf"))

    def _recognize_page_image(self, engine, image) -> tuple[str, list[RecognizedWord]]:
        """Recognizes one rasterized page image, honoring split_spread.
        Returns ("", []) for a fully blank page/pair of halves - never
        invents text for a region with nothing on it. Words are always
        returned in the coordinate space of the full `image` passed in, even
        when split_spread recognizes each half separately - the left half's
        words are shifted right by the split offset, so both text assembly
        and SearchablePDFExporter's word placement stay correct without
        either needing to know split_spread happened."""
        if not self._split_spread:
            return self._recognize_one_region(engine, image)

        width = image.shape[1]
        offset = width // 2
        right_half = image[:, offset:]  # read first - correct order for an RTL book spread
        left_half = image[:, :offset]
        right_text, right_words = self._recognize_one_region(engine, right_half)
        left_text, left_words = self._recognize_one_region(engine, left_half)
        for word in left_words:
            word.x0 += offset
            word.x1 += offset
        text = "\n\n".join(p for p in (right_text, left_text) if p.strip())
        return text, right_words + left_words

    def _recognize_one_region(self, engine, image) -> tuple[str, list[RecognizedWord]]:
        if _is_blank_page(image):
            return "", []
        words = _recognize_with_retry(engine, image)
        self.api_call_made.emit()
        return assemble_text_with_headings(words), words


class UsageCheckWorker(QThread):
    """Queries Cloud Monitoring for the real Vision API request count -
    a network call, so it runs off the GUI thread like OCRWorker does."""

    succeeded = Signal(int)
    failed = Signal(str)

    def __init__(self, credentials_path: str) -> None:
        super().__init__()
        self._credentials_path = credentials_path

    def run(self) -> None:
        try:
            count = get_vision_api_request_count_this_month(self._credentials_path)
            self.succeeded.emit(count)
        except Exception as exc:  # noqa: BLE001 - surface to the GUI, don't crash silently
            self.failed.emit(str(exc))


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Urdu OCR - Google Vision (CLOUD)")
        self.resize(950, 800)

        self._settings = QSettings("UrduOCR", "SimpleGoogleVisionGUI")
        self._input_path: Path | None = None
        self._input_paths: list[Path] | None = None  # multiple-files mode only
        self._input_page_count: int | None = None
        self._worker: OCRWorker | None = None

        cloud_notice = QLabel(
            "This sends page images to Google Cloud Vision for recognition - not offline."
        )
        cloud_notice.setStyleSheet("color: #a03d3d; font-weight: bold;")

        # --- input mode ---
        self.single_file_radio = QRadioButton("Single PDF file")
        self.single_file_radio.setChecked(True)
        self.folder_radio = QRadioButton("Folder of PDFs")
        self.multi_file_radio = QRadioButton("Multiple files...")
        self.single_file_radio.toggled.connect(self._on_input_mode_changed)
        self.folder_radio.toggled.connect(self._on_input_mode_changed)
        self.multi_file_radio.toggled.connect(self._on_input_mode_changed)

        self.input_label = QLabel("Nothing selected")
        input_button = QPushButton("Select...")
        input_button.clicked.connect(self._select_input)

        mode_row = QHBoxLayout()
        mode_row.addWidget(self.single_file_radio)
        mode_row.addWidget(self.folder_radio)
        mode_row.addWidget(self.multi_file_radio)
        mode_row.addStretch(1)

        input_row = QHBoxLayout()
        input_row.addWidget(input_button)
        input_row.addWidget(self.input_label, 1)

        # --- page range (single-file mode only) ---
        self.page_range_label = QLabel("Pages:")
        self.start_page_spin = QSpinBox()
        self.start_page_spin.setMinimum(1)
        self.end_page_spin = QSpinBox()
        self.end_page_spin.setMinimum(1)
        self.start_page_spin.valueChanged.connect(self._on_start_page_changed)
        self.page_count_label = QLabel("")

        page_range_row = QHBoxLayout()
        page_range_row.addWidget(self.page_range_label)
        page_range_row.addWidget(QLabel("from"))
        page_range_row.addWidget(self.start_page_spin)
        page_range_row.addWidget(QLabel("to"))
        page_range_row.addWidget(self.end_page_spin)
        page_range_row.addWidget(self.page_count_label)
        page_range_row.addStretch(1)

        self.split_spread_checkbox = QCheckBox(
            "Each PDF page is a 2-page book spread (scanned side by side) - split and read right half first"
        )
        self.split_spread_checkbox.setChecked(bool(int(self._settings.value("split_spread", 0))))
        self.split_spread_checkbox.toggled.connect(
            lambda checked: self._settings.setValue("split_spread", int(checked))
        )
        self.split_spread_checkbox.setToolTip(
            "Turn this on only if a single PDF page shows two physical book pages side by side "
            "(common when a whole open book was photographed/scanned instead of single sheets). "
            "Doubles the number of Vision API calls per page. Leave off for normal single-page scans - "
            "enabling it on a real single page would incorrectly cut real content in half."
        )

        # --- output folder + filename ---
        self.output_folder_edit = QLineEdit(self._settings.value("output_folder", str(Path.home() / "Documents" / "Urdu OCR Output")))
        output_folder_button = QPushButton("Browse...")
        output_folder_button.clicked.connect(self._select_output_folder)

        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText("output name (single-file mode only)")

        output_folder_row = QHBoxLayout()
        output_folder_row.addWidget(QLabel("Output folder:"))
        output_folder_row.addWidget(self.output_folder_edit, 1)
        output_folder_row.addWidget(output_folder_button)

        output_name_row = QHBoxLayout()
        output_name_row.addWidget(QLabel("Output file name (.txt):"))
        output_name_row.addWidget(self.output_name_edit, 1)

        self.write_docx_checkbox = QCheckBox("Also save as Word (.docx)")
        self.write_docx_checkbox.setChecked(bool(int(self._settings.value("write_docx", 1))))
        self.write_docx_checkbox.toggled.connect(
            lambda checked: self._settings.setValue("write_docx", int(checked))
        )

        self.write_searchable_pdf_checkbox = QCheckBox(
            "Also save as searchable PDF (looks like the original scan, but text is selectable/searchable)"
        )
        self.write_searchable_pdf_checkbox.setChecked(bool(int(self._settings.value("write_searchable_pdf", 0))))
        self.write_searchable_pdf_checkbox.toggled.connect(
            lambda checked: self._settings.setValue("write_searchable_pdf", int(checked))
        )

        # --- quality + credentials ---
        self.quality_combo = QComboBox()
        for label, _dpi in _DPI_OPTIONS:
            self.quality_combo.addItem(label)
        self.quality_combo.setCurrentIndex(int(self._settings.value("quality_index", 1)))  # remembered, default 200 DPI
        self.quality_combo.currentIndexChanged.connect(
            lambda index: self._settings.setValue("quality_index", index)
        )
        self.quality_combo.setToolTip(
            "Higher DPI = larger images, more detail, slower. This range was benchmarked for "
            "processing time; Google Vision's own accuracy sensitivity to DPI has not been "
            "separately measured (only 200 DPI has a real measured result so far)."
        )

        self.credentials_edit = QLineEdit(self._settings.value("credentials_path", ""))
        self.credentials_edit.setPlaceholderText("Path to Google Vision service-account credentials JSON")
        self.credentials_edit.textChanged.connect(self._update_run_enabled)
        credentials_button = QPushButton("Browse...")
        credentials_button.clicked.connect(self._select_credentials)

        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Quality:"))
        quality_row.addWidget(self.quality_combo, 1)

        cred_row = QHBoxLayout()
        cred_row.addWidget(QLabel("Credentials:"))
        cred_row.addWidget(self.credentials_edit, 1)
        cred_row.addWidget(credentials_button)

        # --- usage ---
        self.usage_label = QLabel()
        self.usage_label.setStyleSheet("color: #666666;")
        self._refresh_usage_label()

        self.check_real_usage_button = QPushButton("Check real usage (Google)")
        self.check_real_usage_button.clicked.connect(self._check_real_usage)
        self.real_usage_label = QLabel(
            "Real usage not checked yet - click the button for Google's own confirmed count."
        )
        self.real_usage_label.setStyleSheet("color: #666666;")
        self._usage_check_worker: UsageCheckWorker | None = None

        real_usage_row = QHBoxLayout()
        real_usage_row.addWidget(self.check_real_usage_button)
        real_usage_row.addWidget(self.real_usage_label, 1)

        # --- run / pause / progress / output ---
        self.run_button = QPushButton("Run OCR")
        self.run_button.clicked.connect(self._run_ocr)
        self.run_button.setEnabled(False)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setToolTip(
            "Stops after the current page finishes and saves everything recognized so far. "
            "Click Resume (or just Run OCR again with the same input/output) to continue exactly "
            "where this left off - nothing already recognized is redone or lost."
        )
        self.pause_button.clicked.connect(self._pause_ocr)
        self.pause_button.setEnabled(False)

        run_row = QHBoxLayout()
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.pause_button)

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")

        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.text_view.setLayoutDirection(Qt.RightToLeft)

        layout = QVBoxLayout(self)
        layout.addWidget(cloud_notice)
        layout.addLayout(mode_row)
        layout.addLayout(input_row)
        layout.addLayout(page_range_row)
        layout.addWidget(self.split_spread_checkbox)
        layout.addLayout(output_folder_row)
        layout.addLayout(output_name_row)
        layout.addWidget(self.write_docx_checkbox)
        layout.addWidget(self.write_searchable_pdf_checkbox)
        layout.addLayout(quality_row)
        layout.addLayout(cred_row)
        layout.addWidget(self.usage_label)
        layout.addLayout(real_usage_row)
        layout.addLayout(run_row)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.text_view, 1)

        self._on_input_mode_changed()

    # ---- usage tracking (local estimate only) ----------------------------

    def _usage_key(self) -> str:
        return f"usage_{date.today().strftime('%Y-%m')}"

    def _current_usage(self) -> int:
        return int(self._settings.value(self._usage_key(), 0))

    def _add_usage(self, pages: int) -> None:
        self._settings.setValue(self._usage_key(), self._current_usage() + pages)
        self._refresh_usage_label()

    def _refresh_usage_label(self) -> None:
        used = self._current_usage()
        remaining = max(0, _FREE_TIER_PAGES_PER_MONTH - used)
        self.usage_label.setText(
            f"Estimated usage this month: {used} page(s) sent - ~{remaining} left in the "
            f"{_FREE_TIER_PAGES_PER_MONTH}/month free tier "
            "(counted locally by this app, not synced with Google Cloud billing)."
        )

    # ---- usage tracking (real, from Google Cloud Monitoring) -------------

    def _check_real_usage(self) -> None:
        credentials_path = self.credentials_edit.text().strip()
        if not credentials_path or not Path(credentials_path).exists():
            QMessageBox.warning(self, "No credentials", "Select a valid credentials file first.")
            return
        self.check_real_usage_button.setEnabled(False)
        self.real_usage_label.setText("Checking with Google Cloud Monitoring...")
        self._usage_check_worker = UsageCheckWorker(credentials_path)
        self._usage_check_worker.succeeded.connect(self._on_real_usage_succeeded)
        self._usage_check_worker.failed.connect(self._on_real_usage_failed)
        self._usage_check_worker.start()

    def _on_real_usage_succeeded(self, count: int) -> None:
        self.check_real_usage_button.setEnabled(True)
        self.real_usage_label.setText(
            f"Google-confirmed Vision API requests this month: {count} "
            "(from Cloud Monitoring, not this app's local count)."
        )

    def _on_real_usage_failed(self, message: str) -> None:
        self.check_real_usage_button.setEnabled(True)
        self.real_usage_label.setText("Real usage check failed - see error for details.")
        QMessageBox.critical(
            self,
            "Real usage check failed",
            f"{message}\n\nIf this mentions permission denied, the service account needs the "
            "\"Monitoring Viewer\" IAM role in Cloud Console.",
        )

    # ---- input mode --------------------------------------------------------

    def _on_input_mode_changed(self) -> None:
        self._input_path = None
        self._input_paths = None
        self._input_page_count = None
        self.input_label.setText("Nothing selected")
        is_single = self.single_file_radio.isChecked()
        self.output_name_edit.setEnabled(is_single)
        self.page_range_label.setEnabled(is_single)
        self.start_page_spin.setEnabled(is_single)
        self.end_page_spin.setEnabled(is_single)
        self.page_count_label.setText("")
        self._reset_run_button_to_fresh_job()
        self._update_run_enabled()

    def _last_input_dir(self) -> str:
        return self._settings.value("last_input_dir", "")

    def _remember_input_dir(self, path: Path) -> None:
        directory = path if path.is_dir() else path.parent
        self._settings.setValue("last_input_dir", str(directory))

    def _select_input(self) -> None:
        start_dir = self._last_input_dir()
        if self.single_file_radio.isChecked():
            path_str, _ = QFileDialog.getOpenFileName(self, "Select PDF", start_dir, "PDF files (*.pdf)")
            if path_str:
                self._input_path = Path(path_str)
                self._remember_input_dir(self._input_path)
                self.input_label.setText(self._input_path.name)
                self.output_name_edit.setText(self._input_path.stem + ".txt")
                self._load_page_count()
                self._reset_run_button_to_fresh_job()
        elif self.folder_radio.isChecked():
            path_str = QFileDialog.getExistingDirectory(self, "Select folder of PDFs", start_dir)
            if path_str:
                self._input_path = Path(path_str)
                self._remember_input_dir(self._input_path)
                count = len(list(self._input_path.glob("*.pdf")))
                self.input_label.setText(f"{self._input_path.name} ({count} PDF(s) found)")
                self._reset_run_button_to_fresh_job()
        else:
            path_strs, _ = QFileDialog.getOpenFileNames(self, "Select PDF files", start_dir, "PDF files (*.pdf)")
            if path_strs:
                self._input_paths = [Path(p) for p in path_strs]
                self._remember_input_dir(self._input_paths[0])
                names = ", ".join(p.name for p in self._input_paths[:3])
                if len(self._input_paths) > 3:
                    names += f", +{len(self._input_paths) - 3} more"
                self.input_label.setText(f"{len(self._input_paths)} file(s) selected: {names}")
                self._reset_run_button_to_fresh_job()
        self._update_run_enabled()

    def _load_page_count(self) -> None:
        try:
            with PDFLoader(self._input_path) as loader:
                total = loader.page_count
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not read PDF", str(exc))
            self._input_page_count = None
            self.page_count_label.setText("")
            return

        self._input_page_count = total
        self.start_page_spin.setMaximum(total)
        self.end_page_spin.setMaximum(total)
        self.start_page_spin.setValue(1)
        self.end_page_spin.setValue(total)
        self.page_count_label.setText(f"(of {total} page(s))")

    def _on_start_page_changed(self, value: int) -> None:
        if value > self.end_page_spin.value():
            self.end_page_spin.setValue(value)

    def _select_output_folder(self) -> None:
        path_str = QFileDialog.getExistingDirectory(self, "Select output folder", self.output_folder_edit.text())
        if path_str:
            self.output_folder_edit.setText(path_str)

    def _select_credentials(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Select Google Vision credentials JSON", "", "JSON files (*.json)"
        )
        if path_str:
            self.credentials_edit.setText(path_str)

    def _update_run_enabled(self) -> None:
        has_input = bool(self._input_path) or bool(self._input_paths)
        self.run_button.setEnabled(has_input and bool(self.credentials_edit.text().strip()))

    # ---- cost estimate -----------------------------------------------------

    def _estimate_total_pages(self, pdf_paths: list[Path], page_ranges: list[tuple[int, int] | None]) -> int:
        total = 0
        for pdf_path, page_range in zip(pdf_paths, page_ranges):
            if page_range is not None:
                start, end = page_range
                total += max(0, end - start + 1)
                continue
            try:
                with PDFLoader(pdf_path) as loader:
                    total += loader.page_count
            except Exception:  # noqa: BLE001 - a bad file here just surfaces later as a real error
                pass
        return total

    def _confirm_if_over_free_tier(self, estimated_pages: int) -> bool:
        """Returns True if it's fine to proceed (either within budget, or
        the user confirmed anyway)."""
        used = self._current_usage()
        projected = used + estimated_pages
        if projected <= _FREE_TIER_PAGES_PER_MONTH:
            return True

        over = projected - _FREE_TIER_PAGES_PER_MONTH
        estimated_cost = (over / 1000.0) * _PRICE_PER_1000_PAGES_USD
        reply = QMessageBox.question(
            self,
            "Free tier estimate exceeded",
            f"This job will send an estimated {estimated_pages} page(s). You've already sent "
            f"~{used} this month, which would put you ~{over} page(s) over the "
            f"{_FREE_TIER_PAGES_PER_MONTH}/month free tier - roughly ${estimated_cost:.2f} at "
            f"${_PRICE_PER_1000_PAGES_USD}/1000 pages (this app's own local estimate, not Google's "
            f"actual billing). Continue anyway?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    # ---- run / stop ----------------------------------------------------------

    def _run_ocr(self) -> None:
        credentials_path = self.credentials_edit.text().strip()
        if not credentials_path or not Path(credentials_path).exists():
            QMessageBox.warning(self, "Missing credentials", "Select a valid Google Vision credentials JSON file first.")
            return

        output_folder = Path(self.output_folder_edit.text().strip())
        try:
            output_folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, "Invalid output folder", str(exc))
            return

        if self.single_file_radio.isChecked():
            pdf_paths = [self._input_path]
            name = self.output_name_edit.text().strip() or (self._input_path.stem + ".txt")
            if not name.lower().endswith(".txt"):
                name += ".txt"
            output_paths = [output_folder / name]
            page_ranges: list[tuple[int, int] | None] = [
                (self.start_page_spin.value() - 1, self.end_page_spin.value() - 1)
            ]
        elif self.folder_radio.isChecked():
            pdf_paths = sorted(self._input_path.glob("*.pdf"))
            if not pdf_paths:
                QMessageBox.warning(self, "No PDFs found", "The selected folder has no .pdf files.")
                return
            output_paths = [output_folder / f"{p.stem}.txt" for p in pdf_paths]
            page_ranges = [None] * len(pdf_paths)  # full range per file - a page picker per file isn't practical
        else:
            pdf_paths = self._input_paths
            output_paths = [output_folder / f"{p.stem}.txt" for p in pdf_paths]
            page_ranges = [None] * len(pdf_paths)  # full range per file - a page picker per file isn't practical

        estimated_pages = self._estimate_total_pages(pdf_paths, page_ranges)
        if self.split_spread_checkbox.isChecked():
            estimated_pages *= 2  # each page becomes two API calls (right half + left half)
        if not self._confirm_if_over_free_tier(estimated_pages):
            return

        self._settings.setValue("credentials_path", credentials_path)
        self._settings.setValue("output_folder", str(output_folder))

        dpi = _DPI_OPTIONS[self.quality_combo.currentIndex()][1]

        self.run_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.text_view.clear()
        self.status_label.setText("Starting...")

        self._worker = OCRWorker(
            pdf_paths, output_paths, credentials_path, dpi, page_ranges,
            write_docx=self.write_docx_checkbox.isChecked(),
            write_searchable_pdf=self.write_searchable_pdf_checkbox.isChecked(),
            split_spread=self.split_spread_checkbox.isChecked(),
        )
        self._worker.file_started.connect(self._on_file_started)
        self._worker.page_done.connect(self._on_page_done)
        self._worker.page_recognized.connect(self._on_page_recognized)
        self._worker.api_call_made.connect(lambda: self._add_usage(1))
        self._worker.file_done.connect(self._on_file_done)
        self._worker.file_skipped.connect(self._on_file_skipped)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _pause_ocr(self) -> None:
        if self._worker is not None:
            self.pause_button.setEnabled(False)
            self.status_label.setText("Pausing after the current page...")
            self._worker.request_stop()

    def _reset_run_button_to_fresh_job(self) -> None:
        """Called whenever the input selection changes - a different job,
        not a continuation of whatever was previously paused/failed."""
        self.run_button.setText("Run OCR")

    def _on_file_started(self, pdf_name: str, total_pages: int) -> None:
        self.progress_bar.setMaximum(max(1, total_pages))
        self.progress_bar.setValue(0)
        self.status_label.setText(f"{pdf_name}: starting ({total_pages} page(s))...")
        if self.single_file_radio.isChecked():
            self.text_view.clear()

    def _on_page_done(self, current: int, total: int) -> None:
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_page_recognized(self, current: int, total: int, text: str) -> None:
        # Usage is counted from api_call_made, not inferred here - correct
        # regardless of split-spread mode doubling calls per page.
        if self.single_file_radio.isChecked():
            self.text_view.append(f"--- Page {current}/{total} ---\n{text}\n")

    def _on_file_done(self, pdf_name: str, output_path: str) -> None:
        if not self.single_file_radio.isChecked():
            self.text_view.append(f"Saved: {pdf_name} -> {output_path}\n")

    def _on_file_skipped(self, pdf_name: str) -> None:
        if not self.single_file_radio.isChecked():
            self.text_view.append(f"Skipped (already complete from a previous run): {pdf_name}\n")

    def _on_all_done(self) -> None:
        self.status_label.setText("Done.")
        self.run_button.setText("Run OCR")
        self.run_button.setEnabled(True)
        self.pause_button.setEnabled(False)

    def _on_stopped(self) -> None:
        self.status_label.setText("Paused - pages processed so far were saved. Click Resume to continue.")
        self.run_button.setText("Resume")
        self.run_button.setEnabled(True)
        self.pause_button.setEnabled(False)

    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self, "OCR failed", f"{message}\n\nAny pages already recognized were still saved.")
        self.status_label.setText("Failed - pages processed before the error were saved. Click Resume to continue.")
        self.run_button.setText("Resume")
        self.run_button.setEnabled(True)
        self.pause_button.setEnabled(False)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
