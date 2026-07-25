"""Core data structures shared across rendering, preprocessing, OCR, and export.

Nothing in this module imports Qt or any OCR engine - it is the plain-data
contract the rest of app.core is built around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(slots=True, frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(slots=True)
class OCRWord:
    text: str
    bbox: BBox
    confidence: float  # 0.0-1.0
    engine: str  # "paddleocr" | "tesseract" | "easyocr"
    line_index: int
    is_diacritic: bool = False
    low_confidence: bool = False  # flagged, never auto-corrected


@dataclass(slots=True)
class PageResult:
    page_number: int  # 1-based
    width: int
    height: int
    words: list[OCRWord] = field(default_factory=list)
    chosen_dpi: int = 600
    sharpness_score: float = 0.0
    orientation_angle: float = 0.0
    raw_image_path: Path | None = None
    preprocessed_image_path: Path | None = None
    text: str = ""  # post-processed page text

    @property
    def average_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)

    @property
    def low_confidence_word_count(self) -> int:
        return sum(1 for w in self.words if w.low_confidence)


@dataclass(slots=True)
class DocumentResult:
    source_path: Path
    pages: list[PageResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


@dataclass(slots=True)
class PreprocessConfig:
    """One flag/value per toggleable preprocessing step (GUI spec: every step
    must be individually enable-able). Defaults are conservative for
    Nastaleeq: steps that risk eroding thin joined strokes or small
    diacritic dots (hard thresholding, morphological cleanup) default OFF.
    """

    # Geometry
    orientation_detect: bool = True
    deskew: bool = True
    auto_crop: bool = True
    remove_margins: bool = True

    # Noise / shadow
    noise_classification: bool = True
    denoise: bool = True
    denoise_strength: float = 0.5  # 0.0-1.0
    shadow_removal: bool = True

    # Contrast
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    adaptive_local_contrast: bool = True
    gamma_correction: bool = True
    gamma_value: float = 1.0  # 1.0 = auto-estimate from page brightness

    # Ink / stroke
    stroke_enhancement: bool = True
    ink_enhancement: bool = True
    sharpen: bool = True
    sharpen_amount: float = 0.5  # 0.0-1.0
    faded_text_detection: bool = True

    # Risk-to-dots steps: off by default, available per document/benchmark
    morphological_cleanup: bool = False
    adaptive_threshold: bool = False
    threshold_method: str = "adaptive_gaussian"  # "adaptive_gaussian" | "adaptive_mean" | "otsu"

    # Resize / super-resolution
    resize_if_needed: bool = True
    min_dimension_px: int = 1600
    super_resolution: bool = False
    super_resolution_scale: int = 2  # 2 | 3 | 4 (FSRCNN model variant)

    # Dot/diacritic protection - consulted by denoise/morphology/threshold steps
    preserve_diacritics: bool = True


@dataclass(slots=True)
class OCRConfig:
    engine_confidence_threshold: float = 0.75
    lang_paddle: str = "ur"
    lang_tesseract: str = "urd"
    use_tesseract_fallback: bool = True
    use_easyocr_comparison: bool = False

    # Adaptive multi-DPI rendering
    adaptive_dpi: bool = True
    base_dpi: int = 600
    dpi_escalation_steps: tuple[int, ...] = (600, 900, 1200)
    sharpness_threshold: float = 100.0  # Laplacian variance; below this, escalate DPI

    # Escalating to 900/1200 DPI recovers detail from a blurry scan, but a
    # normal page at 1200 DPI is 100+ megapixels - there is no accuracy
    # benefit to running the full CV preprocessing pipeline and OCR at that
    # resolution, only cost. The chosen render is downscaled to this working
    # resolution (long side, in px) right after DPI escalation picks its
    # winner, before any preprocessing runs.
    max_working_dimension_px: int = 3500

    # Multi-scale OCR: run OCR on every DPI variant instead of just the chosen one
    multiscale_ocr: bool = False


@dataclass(slots=True)
class BenchmarkVariant:
    name: str
    dpi: int
    preprocess: PreprocessConfig


@dataclass(slots=True)
class BenchmarkRunResult:
    variant_name: str
    dpi: int
    average_confidence: float
    word_count: int
    elapsed_seconds: float
    preprocess: PreprocessConfig  # carried along so "apply this pipeline" needs no lookup
