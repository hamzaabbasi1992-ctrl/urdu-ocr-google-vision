"""PaddleOCR engine wrapper - the primary OCR engine (Step 4).

PaddleOCR's Python API changed shape between the 2.x `.ocr()` call and the
3.x `.predict()` pipeline API. This wrapper supports both so it keeps
working across the version actually installed, and logs a clear error if a
future version changes the result shape again rather than silently
returning nothing.

`enable_mkldnn=False` works around a real bug (not a version-shape quirk):
PaddlePaddle 3.3.1's PIR execution mode combined with its oneDNN CPU
acceleration path raises `NotImplementedError:
ConvertPirAttribute2RuntimeAttribute not support
[pir::ArrayAttribute<pir::DoubleAttribute>]` on PP-OCRv5's detection model -
confirmed as a known upstream issue (PaddlePaddle/Paddle#77340,
PaddleOCR#18162) on exactly this paddlepaddle/paddleocr version pair, with
no fix released yet. Disabling oneDNN avoids the broken code path entirely;
since this machine has no GPU anyway, oneDNN was the only CPU acceleration
being lost.
"""

from __future__ import annotations

import logging

import numpy as np

from app.core.models import BBox, OCRWord
from app.core.ocr.engine_base import assign_line_indices, polygon_to_bbox

_LOGGER = logging.getLogger("urdu_ocr.ocr.paddle")


class PaddleEngine:
    name = "paddleocr"

    def __init__(self, lang: str = "ur", use_angle_cls: bool = True) -> None:
        from paddleocr import PaddleOCR

        self._lang = lang
        # PaddleOCR's constructor kwargs changed across the 2.x -> 3.x
        # pipeline-API rewrite (use_angle_cls/show_log -> use_textline_orientation,
        # or dropped outright), so try each known shape before falling back to
        # the bare minimum that every version accepts.
        for kwargs in (
            {"lang": lang, "use_textline_orientation": use_angle_cls, "enable_mkldnn": False},
            {"lang": lang, "use_textline_orientation": use_angle_cls},
            {"lang": lang, "use_angle_cls": use_angle_cls, "show_log": False, "enable_mkldnn": False},
            {"lang": lang, "use_angle_cls": use_angle_cls, "show_log": False},
            {"lang": lang, "enable_mkldnn": False},
            {"lang": lang},
        ):
            try:
                self._ocr = PaddleOCR(**kwargs)
                break
            except (TypeError, ValueError):
                continue
        else:
            raise RuntimeError("Could not construct PaddleOCR with any known argument set")

    def run(self, image: np.ndarray) -> list[OCRWord]:
        bgr = np.stack([image, image, image], axis=-1) if image.ndim == 2 else image

        if hasattr(self._ocr, "predict"):
            raw = self._ocr.predict(bgr)
        else:
            raw = self._ocr.ocr(bgr, cls=True)

        words = self._parse(raw)
        assign_line_indices(words)
        return words

    def _parse(self, raw) -> list[OCRWord]:
        words: list[OCRWord] = []
        try:
            for page_result in raw:
                words.extend(self._parse_page(page_result))
        except Exception as exc:  # noqa: BLE001 - result-shape drift must be visible, not a silent empty page
            _LOGGER.error("Could not parse PaddleOCR result (API shape mismatch?): %s", exc, exc_info=True)
        return words

    def _parse_page(self, page_result) -> list[OCRWord]:
        words: list[OCRWord] = []

        # Modern (3.x) dict-like OCRResult: rec_texts / rec_scores / rec_polys|dt_polys
        texts = _get(page_result, "rec_texts")
        if texts is not None:
            scores = _get(page_result, "rec_scores") or [1.0] * len(texts)
            polys = _get(page_result, "rec_polys") or _get(page_result, "dt_polys") or []
            for text, score, poly in zip(texts, scores, polys):
                if not text.strip():
                    continue
                words.append(
                    OCRWord(text=text, bbox=polygon_to_bbox(poly), confidence=float(score), engine=self.name, line_index=0)
                )
            return words

        # Legacy 2.x .ocr() shape: list of [box_points, (text, score)]
        if page_result:
            for line in page_result:
                try:
                    box_points, (text, score) = line
                except (ValueError, TypeError):
                    continue
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
        return words


def _get(obj, key: str):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
