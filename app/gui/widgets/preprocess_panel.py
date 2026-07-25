"""Every preprocessing step from PreprocessConfig, individually toggleable
from the GUI (spec requirement). Grouped into boxes that roughly follow the
pipeline's execution order.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.models import PreprocessConfig


class _ToggleSlider(QWidget):
    """A checkbox plus an optional slider on one row, sharing enabled state."""

    def __init__(self, label: str, tooltip: str = "", with_slider: bool = False, slider_range=(0, 100)) -> None:
        super().__init__()
        self.checkbox = QCheckBox(label)
        if tooltip:
            self.checkbox.setToolTip(tooltip)
        self.slider: QSlider | None = None
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow(self.checkbox)
        if with_slider:
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(*slider_range)
            self.checkbox.toggled.connect(self.slider.setEnabled)
            layout.addRow(self.slider)


class PreprocessPanel(QWidget):
    config_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.orientation_detect = QCheckBox("Detect page orientation automatically")
        self.deskew = QCheckBox("Deskew")
        self.auto_crop = QCheckBox("Automatic page cropping")
        self.remove_margins = QCheckBox("Automatic margin removal")

        self.noise_classification = QCheckBox("Noise classification (choose denoise method automatically)")
        self.denoise = _ToggleSlider("Remove background noise / denoise", with_slider=True)
        self.shadow_removal = QCheckBox("Shadow removal")

        self.clahe = _ToggleSlider(
            "CLAHE contrast enhancement", with_slider=True, slider_range=(10, 40)
        )
        self.adaptive_local_contrast = QCheckBox("Adaptive local contrast")
        self.gamma_correction = _ToggleSlider(
            "Gamma correction (center = auto)", with_slider=True, slider_range=(50, 250)
        )

        self.stroke_enhancement = QCheckBox("Stroke enhancement (preserve Nastaleeq curves)")
        self.ink_enhancement = QCheckBox("Ink enhancement")
        self.sharpen = _ToggleSlider("Sharpen", with_slider=True)
        self.faded_text_detection = QCheckBox("Detect and boost faded text")
        self.preserve_diacritics = QCheckBox("Preserve zabar/zer/pesh and other diacritics (recommended)")

        self.morphological_cleanup = QCheckBox("Morphological cleanup (risk: may erase small dots)")
        self.adaptive_threshold = QCheckBox("Adaptive threshold / binarize (risk: may break thin joins)")
        self.threshold_method = QComboBox()
        self.threshold_method.addItems(["adaptive_gaussian", "adaptive_mean", "otsu"])

        self.resize_if_needed = QCheckBox("Resize if below minimum dimension")
        self.min_dimension_px = QSpinBox()
        self.min_dimension_px.setRange(500, 6000)
        self.min_dimension_px.setSingleStep(100)

        self.super_resolution = QCheckBox("Super-resolution before OCR (slow on CPU)")
        self.super_resolution_scale = QComboBox()
        self.super_resolution_scale.addItems(["2", "3", "4"])

        self._build_layout()
        self._load_defaults(PreprocessConfig())
        self._wire_change_signal()

    def _group(self, title: str, widgets: list[QWidget]) -> QGroupBox:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        for w in widgets:
            layout.addWidget(w)
        return box

    def _build_layout(self) -> None:
        content = QWidget()
        outer = QVBoxLayout(content)

        outer.addWidget(self._group(
            "Geometry", [self.orientation_detect, self.deskew, self.auto_crop, self.remove_margins]
        ))
        outer.addWidget(self._group(
            "Noise && Shadow", [self.noise_classification, self.denoise, self.shadow_removal]
        ))
        outer.addWidget(self._group(
            "Contrast", [self.clahe, self.adaptive_local_contrast, self.gamma_correction]
        ))
        outer.addWidget(self._group(
            "Ink && Sharpening",
            [self.stroke_enhancement, self.ink_enhancement, self.sharpen,
             self.faded_text_detection, self.preserve_diacritics],
        ))

        threshold_row = QWidget()
        threshold_layout = QFormLayout(threshold_row)
        threshold_layout.addRow(self.adaptive_threshold)
        threshold_layout.addRow(QLabel("Method:"), self.threshold_method)
        outer.addWidget(self._group("Risky steps (off by default)", [self.morphological_cleanup, threshold_row]))

        resize_row = QWidget()
        resize_layout = QFormLayout(resize_row)
        resize_layout.addRow(self.resize_if_needed)
        resize_layout.addRow(QLabel("Minimum dimension (px):"), self.min_dimension_px)
        sr_row = QWidget()
        sr_layout = QFormLayout(sr_row)
        sr_layout.addRow(self.super_resolution)
        sr_layout.addRow(QLabel("Scale:"), self.super_resolution_scale)
        outer.addWidget(self._group("Resize && Super-Resolution", [resize_row, sr_row]))
        outer.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

    def _wire_change_signal(self) -> None:
        checkboxes = [
            self.orientation_detect, self.deskew, self.auto_crop, self.remove_margins,
            self.noise_classification, self.denoise.checkbox, self.shadow_removal,
            self.clahe.checkbox, self.adaptive_local_contrast, self.gamma_correction.checkbox,
            self.stroke_enhancement, self.ink_enhancement, self.sharpen.checkbox,
            self.faded_text_detection, self.preserve_diacritics,
            self.morphological_cleanup, self.adaptive_threshold,
            self.resize_if_needed, self.super_resolution,
        ]
        for cb in checkboxes:
            cb.toggled.connect(self.config_changed.emit)

    def _load_defaults(self, config: PreprocessConfig) -> None:
        self.load_config(config)

    def load_config(self, config: PreprocessConfig) -> None:
        self.orientation_detect.setChecked(config.orientation_detect)
        self.deskew.setChecked(config.deskew)
        self.auto_crop.setChecked(config.auto_crop)
        self.remove_margins.setChecked(config.remove_margins)

        self.noise_classification.setChecked(config.noise_classification)
        self.denoise.checkbox.setChecked(config.denoise)
        self.denoise.slider.setValue(int(config.denoise_strength * 100))
        self.shadow_removal.setChecked(config.shadow_removal)

        self.clahe.checkbox.setChecked(config.clahe)
        self.clahe.slider.setValue(int(config.clahe_clip_limit * 10))
        self.adaptive_local_contrast.setChecked(config.adaptive_local_contrast)
        self.gamma_correction.checkbox.setChecked(config.gamma_correction)
        self.gamma_correction.slider.setValue(int(config.gamma_value * 100))

        self.stroke_enhancement.setChecked(config.stroke_enhancement)
        self.ink_enhancement.setChecked(config.ink_enhancement)
        self.sharpen.checkbox.setChecked(config.sharpen)
        self.sharpen.slider.setValue(int(config.sharpen_amount * 100))
        self.faded_text_detection.setChecked(config.faded_text_detection)
        self.preserve_diacritics.setChecked(config.preserve_diacritics)

        self.morphological_cleanup.setChecked(config.morphological_cleanup)
        self.adaptive_threshold.setChecked(config.adaptive_threshold)
        idx = self.threshold_method.findText(config.threshold_method)
        if idx >= 0:
            self.threshold_method.setCurrentIndex(idx)

        self.resize_if_needed.setChecked(config.resize_if_needed)
        self.min_dimension_px.setValue(config.min_dimension_px)

        self.super_resolution.setChecked(config.super_resolution)
        scale_idx = self.super_resolution_scale.findText(str(config.super_resolution_scale))
        if scale_idx >= 0:
            self.super_resolution_scale.setCurrentIndex(scale_idx)

    def get_config(self) -> PreprocessConfig:
        return PreprocessConfig(
            orientation_detect=self.orientation_detect.isChecked(),
            deskew=self.deskew.isChecked(),
            auto_crop=self.auto_crop.isChecked(),
            remove_margins=self.remove_margins.isChecked(),
            noise_classification=self.noise_classification.isChecked(),
            denoise=self.denoise.checkbox.isChecked(),
            denoise_strength=self.denoise.slider.value() / 100.0,
            shadow_removal=self.shadow_removal.isChecked(),
            clahe=self.clahe.checkbox.isChecked(),
            clahe_clip_limit=self.clahe.slider.value() / 10.0,
            adaptive_local_contrast=self.adaptive_local_contrast.isChecked(),
            gamma_correction=self.gamma_correction.checkbox.isChecked(),
            gamma_value=self.gamma_correction.slider.value() / 100.0,
            stroke_enhancement=self.stroke_enhancement.isChecked(),
            ink_enhancement=self.ink_enhancement.isChecked(),
            sharpen=self.sharpen.checkbox.isChecked(),
            sharpen_amount=self.sharpen.slider.value() / 100.0,
            faded_text_detection=self.faded_text_detection.isChecked(),
            morphological_cleanup=self.morphological_cleanup.isChecked(),
            adaptive_threshold=self.adaptive_threshold.isChecked(),
            threshold_method=self.threshold_method.currentText(),
            resize_if_needed=self.resize_if_needed.isChecked(),
            min_dimension_px=self.min_dimension_px.value(),
            super_resolution=self.super_resolution.isChecked(),
            super_resolution_scale=int(self.super_resolution_scale.currentText()),
            preserve_diacritics=self.preserve_diacritics.isChecked(),
        )
