"""OCR engine and adaptive-DPI-rendering settings."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.core.models import OCRConfig


class OCRSettingsPanel(QWidget):
    config_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.confidence_threshold = QSlider(Qt.Horizontal)
        self.confidence_threshold.setRange(0, 100)
        self.confidence_threshold.setValue(75)
        self.confidence_label = QLabel("0.75")
        self.confidence_threshold.valueChanged.connect(
            lambda v: self.confidence_label.setText(f"{v / 100:.2f}")
        )

        self.use_tesseract_fallback = QCheckBox("Use Tesseract as fallback when confidence is low")
        self.use_easyocr_comparison = QCheckBox("Also run EasyOCR for side-by-side comparison (Arabic model - approximate)")

        self.adaptive_dpi = QCheckBox("Adaptive multi-DPI rendering (600 -> 900 -> 1200 if soft)")
        self.sharpness_threshold = QSlider(Qt.Horizontal)
        self.sharpness_threshold.setRange(10, 500)
        self.sharpness_threshold.setValue(100)
        self.sharpness_label = QLabel("100")
        self.sharpness_threshold.valueChanged.connect(lambda v: self.sharpness_label.setText(str(v)))

        self.multiscale_ocr = QCheckBox("Multi-scale OCR (run OCR at every tried DPI, slower)")

        engine_box = QGroupBox("Engine arbitration")
        engine_layout = QFormLayout(engine_box)
        engine_layout.addRow("Confidence threshold:", self._row(self.confidence_threshold, self.confidence_label))
        engine_layout.addRow(self.use_tesseract_fallback)
        engine_layout.addRow(self.use_easyocr_comparison)

        dpi_box = QGroupBox("Rendering")
        dpi_layout = QFormLayout(dpi_box)
        dpi_layout.addRow(self.adaptive_dpi)
        dpi_layout.addRow("Sharpness threshold:", self._row(self.sharpness_threshold, self.sharpness_label))
        dpi_layout.addRow(self.multiscale_ocr)

        layout = QVBoxLayout(self)
        layout.addWidget(engine_box)
        layout.addWidget(dpi_box)
        layout.addStretch(1)

        for w in (self.use_tesseract_fallback, self.use_easyocr_comparison, self.adaptive_dpi, self.multiscale_ocr):
            w.toggled.connect(self.config_changed.emit)
        self.confidence_threshold.valueChanged.connect(self.config_changed.emit)
        self.sharpness_threshold.valueChanged.connect(self.config_changed.emit)

        self.load_config(OCRConfig())

    def _row(self, *widgets: QWidget) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        for w in widgets:
            layout.addWidget(w)
        return container

    def load_config(self, config: OCRConfig) -> None:
        self.confidence_threshold.setValue(int(config.engine_confidence_threshold * 100))
        self.use_tesseract_fallback.setChecked(config.use_tesseract_fallback)
        self.use_easyocr_comparison.setChecked(config.use_easyocr_comparison)
        self.adaptive_dpi.setChecked(config.adaptive_dpi)
        self.sharpness_threshold.setValue(int(config.sharpness_threshold))
        self.multiscale_ocr.setChecked(config.multiscale_ocr)

    def get_config(self) -> OCRConfig:
        return OCRConfig(
            engine_confidence_threshold=self.confidence_threshold.value() / 100.0,
            use_tesseract_fallback=self.use_tesseract_fallback.isChecked(),
            use_easyocr_comparison=self.use_easyocr_comparison.isChecked(),
            adaptive_dpi=self.adaptive_dpi.isChecked(),
            sharpness_threshold=float(self.sharpness_threshold.value()),
            multiscale_ocr=self.multiscale_ocr.isChecked(),
        )
