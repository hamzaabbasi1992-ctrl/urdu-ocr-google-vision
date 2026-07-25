"""Shows the original render next to the preprocessed result for one page,
with page navigation. Preview runs synchronously against a single page (not
a background worker) - it's a bounded, one-page operation, unlike batch/
benchmark runs which can touch hundreds of pages and always run off-thread.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.models import OCRConfig, PreprocessConfig
from app.core.pdf_render import open_pdf, page_count, render_page_adaptive
from app.core.preprocess.pipeline import run_pipeline

_LOGGER = logging.getLogger("urdu_ocr.gui.page_preview")


def numpy_to_pixmap(image: np.ndarray) -> QPixmap:
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    qimage = QImage(image.data, w, h, image.strides[0], QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimage.copy())


class PagePreviewWidget(QWidget):
    preview_updated = Signal(object, object)  # PreprocessDebugInfo, chosen_dpi

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pdf_path: Path | None = None

        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.preview_button = QPushButton("Preview This Page")
        self.preview_button.clicked.connect(self._on_preview)
        self.status_label = QLabel("No PDF loaded")

        nav_row = QHBoxLayout()
        nav_row.addWidget(QLabel("Page:"))
        nav_row.addWidget(self.page_spin)
        nav_row.addWidget(self.preview_button)
        nav_row.addWidget(self.status_label)
        nav_row.addStretch(1)

        self.original_label = QLabel("Original")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.processed_label = QLabel("Preprocessed")
        self.processed_label.setAlignment(Qt.AlignCenter)

        original_scroll = QScrollArea()
        original_scroll.setWidgetResizable(True)
        original_scroll.setWidget(self.original_label)
        processed_scroll = QScrollArea()
        processed_scroll.setWidgetResizable(True)
        processed_scroll.setWidget(self.processed_label)

        images_row = QHBoxLayout()
        images_row.addWidget(original_scroll)
        images_row.addWidget(processed_scroll)

        layout = QVBoxLayout(self)
        layout.addLayout(nav_row)
        layout.addLayout(images_row)

        self._get_preprocess_config = None
        self._get_ocr_config = None

    def bind_config_providers(self, get_preprocess_config, get_ocr_config) -> None:
        self._get_preprocess_config = get_preprocess_config
        self._get_ocr_config = get_ocr_config

    def set_pdf(self, pdf_path: Path | None) -> None:
        self._pdf_path = pdf_path
        if pdf_path is None:
            self.status_label.setText("No PDF loaded")
            self.page_spin.setMaximum(1)
            return
        try:
            count = page_count(pdf_path)
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"Could not open PDF: {exc}")
            return
        self.page_spin.setMaximum(max(1, count))
        self.status_label.setText(f"{pdf_path.name} - {count} page(s)")

    def current_page_index(self) -> int:
        return self.page_spin.value() - 1

    def _on_preview(self) -> None:
        if self._pdf_path is None or self._get_preprocess_config is None:
            return
        preprocess_config: PreprocessConfig = self._get_preprocess_config()
        ocr_config: OCRConfig = self._get_ocr_config()
        page_index = self.current_page_index()

        self.status_label.setText("Rendering...")
        try:
            with open_pdf(self._pdf_path) as doc:
                from app.core.pdf_render import new_document_temp_dir, cleanup_document_temp_dir

                temp_dir = new_document_temp_dir(self._pdf_path)
                try:
                    adaptive = render_page_adaptive(doc, page_index, ocr_config, temp_dir)
                    processed, debug = run_pipeline(adaptive.chosen.image, preprocess_config, adaptive.chosen.dpi)
                finally:
                    cleanup_document_temp_dir(temp_dir)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error("Preview failed: %s", exc, exc_info=True)
            self.status_label.setText(f"Preview failed: {exc}")
            return

        self.original_label.setPixmap(numpy_to_pixmap(adaptive.chosen.image))
        self.processed_label.setPixmap(numpy_to_pixmap(processed))
        self.status_label.setText(
            f"{self._pdf_path.name} - page {page_index + 1} - {adaptive.chosen.dpi} DPI "
            f"(sharpness {adaptive.sharpness_score:.0f}) - steps: {', '.join(debug.applied_steps) or 'none'}"
        )
        self.preview_updated.emit(debug, adaptive.chosen.dpi)
