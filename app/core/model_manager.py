"""Thread-safe cache of loaded OCR engines, so switching pages/files in a
batch doesn't reload PaddleOCR/EasyOCR's model weights each time - mirrors
the sibling transcriber app's ModelManager pattern.
"""

from __future__ import annotations

import threading

from app.core.models import OCRConfig
from app.core.ocr.arbiter import Arbiter
from app.core.ocr.easyocr_engine import EasyOCREngine


class ModelManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cached_key: tuple | None = None
        self._cached_arbiter: Arbiter | None = None
        self._easyocr: EasyOCREngine | None = None

    def get_arbiter(self, ocr_config: OCRConfig) -> Arbiter:
        key = (
            ocr_config.lang_paddle,
            ocr_config.lang_tesseract,
            ocr_config.use_tesseract_fallback,
            ocr_config.engine_confidence_threshold,
        )
        with self._lock:
            if self._cached_arbiter is not None and self._cached_key == key:
                return self._cached_arbiter
            arbiter = Arbiter(ocr_config)
            self._cached_arbiter = arbiter
            self._cached_key = key
            return arbiter

    def get_easyocr(self) -> EasyOCREngine:
        with self._lock:
            if self._easyocr is None:
                self._easyocr = EasyOCREngine()
            return self._easyocr
