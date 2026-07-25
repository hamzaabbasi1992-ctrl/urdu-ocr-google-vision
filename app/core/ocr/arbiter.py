"""Step 4: run PaddleOCR first; for any region below the confidence
threshold, run Tesseract over the same page and take whichever engine was
more confident for that region. Confidence is stored per word either way.

Bbox granularity differs between engines (Tesseract's word segmentation
doesn't line up 1:1 with PaddleOCR's detected text regions for cursive
Urdu), so matching is done by bounding-box overlap (IoU) rather than
assuming a word-for-word correspondence - the best-overlapping Tesseract
region is the candidate replacement for a given PaddleOCR region.

This 2-engine arbiter is the pre-PROJECT_SPEC.md baseline design, superseded
by the Layer 7 Cross-Engine Fusion architecture (see PROJECT_SPEC.md
Section 5) once that migration happens module-by-module.
"""

from __future__ import annotations

import logging

import numpy as np

from app.core.models import OCRConfig, OCRWord
from app.core.ocr.engine_base import bbox_iou
from app.core.ocr.paddle_engine import PaddleEngine
from app.core.ocr.tesseract_engine import TesseractEngine

_LOGGER = logging.getLogger("urdu_ocr.ocr.arbiter")


def _best_overlap(word: OCRWord, candidates: list[OCRWord]) -> OCRWord | None:
    best, best_iou = None, 0.2  # minimum overlap to consider it "the same region"
    for candidate in candidates:
        iou = bbox_iou(word.bbox, candidate.bbox)
        if iou > best_iou:
            best, best_iou = candidate, iou
    return best


class Arbiter:
    def __init__(self, ocr_config: OCRConfig) -> None:
        self._config = ocr_config
        self._paddle: PaddleEngine | None = None
        self._tesseract: TesseractEngine | None = None

    def run(self, image: np.ndarray) -> list[OCRWord]:
        if self._paddle is None:
            self._paddle = PaddleEngine(lang=self._config.lang_paddle)
        threshold = self._config.engine_confidence_threshold
        paddle_words = self._paddle.run(image)

        needs_fallback = any(w.confidence < threshold for w in paddle_words) or not paddle_words
        if needs_fallback and self._config.use_tesseract_fallback:
            if self._tesseract is None:
                self._tesseract = TesseractEngine(lang=self._config.lang_tesseract)
            tesseract_words = self._tesseract.run(image)
            _LOGGER.debug(
                "Low-confidence page: paddle=%d words, tesseract fallback=%d words",
                len(paddle_words), len(tesseract_words),
            )
            merged = self._merge(paddle_words, tesseract_words)
        else:
            merged = paddle_words

        for word in merged:
            word.low_confidence = word.confidence < threshold
        return merged

    def _merge(self, paddle_words: list[OCRWord], tesseract_words: list[OCRWord]) -> list[OCRWord]:
        result: list[OCRWord] = []
        for word in paddle_words:
            if word.confidence >= self._config.engine_confidence_threshold:
                result.append(word)
                continue
            candidate = _best_overlap(word, tesseract_words)
            if candidate is not None and candidate.confidence > word.confidence:
                result.append(
                    OCRWord(
                        text=candidate.text,
                        bbox=word.bbox,
                        confidence=candidate.confidence,
                        engine=candidate.engine,
                        line_index=word.line_index,
                        is_diacritic=word.is_diacritic,
                    )
                )
            else:
                result.append(word)

        # Tesseract regions with no PaddleOCR counterpart at all (PaddleOCR missed the region)
        for t_word in tesseract_words:
            if _best_overlap(t_word, paddle_words) is None:
                result.append(t_word)

        return result
