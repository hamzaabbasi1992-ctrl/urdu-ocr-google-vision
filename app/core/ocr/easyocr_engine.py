"""EasyOCR wrapper - comparison-only, never part of the arbiter's decision.

EasyOCR has no dedicated Urdu model; it runs with its Arabic ('ar') model as
the closest available reference. The GUI must label this output clearly as
"Arabic model (approximate)" so it isn't mistaken for an Urdu-tuned result.
"""

from __future__ import annotations

import numpy as np

from app.core.models import BBox, OCRWord
from app.core.ocr.engine_base import assign_line_indices, polygon_to_bbox

_reader_cache = None


class EasyOCREngine:
    name = "easyocr"

    def __init__(self) -> None:
        global _reader_cache
        if _reader_cache is None:
            import easyocr

            _reader_cache = easyocr.Reader(["ar"], gpu=False)
        self._reader = _reader_cache

    def run(self, image: np.ndarray) -> list[OCRWord]:
        results = self._reader.readtext(image)
        words: list[OCRWord] = []
        for box_points, text, score in results:
            if not text.strip():
                continue
            words.append(
                OCRWord(
                    text=text,
                    bbox=polygon_to_bbox(box_points),
                    confidence=float(score),
                    engine=self.name,
                    line_index=0,
                )
            )
        assign_line_indices(words)
        return words
