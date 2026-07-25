"""Tesseract engine wrapper - fallback/arbiter partner for PaddleOCR (Step 4).

Requires the system Tesseract install to have `urd.traineddata` (checked at
first run by tools/check_tesseract_urdu.ps1).
"""

from __future__ import annotations

import numpy as np

from app.core.models import BBox, OCRWord
from app.core.ocr.engine_base import assign_line_indices


class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang: str = "urd") -> None:
        self._lang = lang

    def run(self, image: np.ndarray) -> list[OCRWord]:
        import pytesseract
        from pytesseract import Output

        data = pytesseract.image_to_data(image, lang=self._lang, output_type=Output.DICT)

        words: list[OCRWord] = []
        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue
            conf = data["conf"][i]
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                continue
            if conf < 0:
                continue

            left, top = data["left"][i], data["top"][i]
            width, height = data["width"][i], data["height"][i]
            bbox = BBox(x0=float(left), y0=float(top), x1=float(left + width), y1=float(top + height))
            words.append(OCRWord(text=text, bbox=bbox, confidence=conf / 100.0, engine=self.name, line_index=0))

        assign_line_indices(words)
        return words
