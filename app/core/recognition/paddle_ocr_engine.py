"""Layer 6 - Recognition Engines: PaddleOCREngine.

Single responsibility: recognize text via PaddleOCR, returning recognized
words in reading order (grouped by line, right-to-left within each line,
per Urdu's writing direction). Reading-order assembly is included here
rather than as a separate module - per the directive to minimize the
codebase, this is treated as part of "turning PaddleOCR's raw per-region
output into usable text," not a distinct responsibility, since PaddleOCR's
own output has no implicit reading order at all (unordered per-region
detections) and is useless for a CER/WER benchmark without it - a scrambled
reading order would produce a terrible CER even with perfect per-word
recognition.

The version-fallback constructor, dict/legacy result-shape parsing, and
`enable_mkldnn=False` fix are ported unchanged from the old
`app/core/ocr/paddle_engine.py` (see that file's docstring for why each is
needed - confirmed real bugs on this exact paddlepaddle/paddleocr version
pair, not speculative defensiveness).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

_LOGGER = logging.getLogger("urdu_ocr.recognition.paddle_ocr_engine")

# PaddleOCR has no dedicated Urdu recognition model; lang="ur" auto-selects
# this Arabic-script one (confirmed from PaddleOCR's own model-selection
# log output). Mapped explicitly here because pinning
# text_detection_model_name forces pinning text_recognition_model_name too
# (see PaddleOCREngine.__init__).
_RECOGNITION_MODEL_BY_LANG = {
    "ur": "arabic_PP-OCRv5_mobile_rec",
    "ar": "arabic_PP-OCRv5_mobile_rec",
}


def _recognition_model_for_lang(lang: str) -> str | None:
    return _RECOGNITION_MODEL_BY_LANG.get(lang)


@dataclass(slots=True)
class RecognizedWord:
    text: str
    confidence: float
    x0: float
    y0: float
    x1: float
    y1: float
    line_index: int = 0


class PaddleOCREngine:
    def __init__(self, lang: str = "ur") -> None:
        from paddleocr import PaddleOCR

        # text_detection_model_name="PP-OCRv5_mobile_det" works around a
        # real native crash (Windows access violation / segfault inside
        # PaddlePaddle's inference runtime, confirmed as a known class of
        # PaddleOCR-3.x-on-CPU/Windows issue) that occurs in the heavier
        # "server" detection model on real page-sized images - the tiny
        # test image in this module's own unit test didn't trigger it, only
        # a real ~3400x4400 page did.
        #
        # PaddleOCR's warning "`lang` and `ocr_version` will be ignored when
        # model names ... are not None" is not cosmetic: passing
        # text_detection_model_name alone silently drops lang-based
        # recognition-model selection too, falling back to a generic
        # (non-Arabic) recognition model - confirmed by an actual run
        # (CER 0.98, WER 1.0, garbled output) before this was caught. Both
        # model names must be pinned together, or neither.
        for kwargs in (
            {
                "use_textline_orientation": True,
                "enable_mkldnn": False,
                "text_detection_model_name": "PP-OCRv5_mobile_det",
                "text_recognition_model_name": _recognition_model_for_lang(lang),
            },
            {"lang": lang, "use_textline_orientation": True, "enable_mkldnn": False},
            {"lang": lang, "use_textline_orientation": True},
            {"lang": lang, "use_angle_cls": True, "show_log": False, "enable_mkldnn": False},
            {"lang": lang, "use_angle_cls": True, "show_log": False},
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

    def recognize(self, image: np.ndarray) -> list[RecognizedWord]:
        bgr = np.stack([image, image, image], axis=-1) if image.ndim == 2 else image

        if hasattr(self._ocr, "predict"):
            raw = self._ocr.predict(bgr)
        else:
            raw = self._ocr.ocr(bgr, cls=True)

        words = parse_paddleocr_result(raw)
        assign_reading_order(words)
        return words


def parse_paddleocr_result(raw) -> list[RecognizedWord]:
    """Pulled out as a free function so it can be unit-tested against fake
    result objects without constructing a real (slow, model-loading)
    PaddleOCREngine."""
    words: list[RecognizedWord] = []
    try:
        for page_result in raw:
            words.extend(_parse_page(page_result))
    except Exception as exc:  # noqa: BLE001 - result-shape drift must be visible, not a silent empty page
        _LOGGER.error("Could not parse PaddleOCR result (API shape mismatch?): %s", exc, exc_info=True)
    return words


def _parse_page(page_result) -> list[RecognizedWord]:
    words: list[RecognizedWord] = []

    # Modern (3.x) dict-like OCRResult: rec_texts / rec_scores / rec_polys|dt_polys
    texts = _get(page_result, "rec_texts")
    if texts is not None:
        scores = _get(page_result, "rec_scores") or [1.0] * len(texts)
        polys = _get(page_result, "rec_polys") or _get(page_result, "dt_polys") or []
        for text, score, poly in zip(texts, scores, polys):
            if not text.strip():
                continue
            x0, y0, x1, y1 = _bbox_from_poly(poly)
            words.append(RecognizedWord(text=text, confidence=float(score), x0=x0, y0=y0, x1=x1, y1=y1))
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
            x0, y0, x1, y1 = _bbox_from_poly(box_points)
            words.append(RecognizedWord(text=text, confidence=float(score), x0=x0, y0=y0, x1=x1, y1=y1))
    return words


def assign_reading_order(words: list[RecognizedWord]) -> None:
    """Groups words into lines by vertical center (tolerant of a slightly
    stepped/diagonal baseline), then sorts right-to-left within each line.
    Mutates in place."""
    if not words:
        return

    heights = [w.y1 - w.y0 for w in words]
    median_height = sorted(heights)[len(heights) // 2] or 1.0
    tolerance = median_height * 0.6

    ordered = sorted(words, key=lambda w: (w.y0 + w.y1) / 2)
    line_index = 0
    line_center = (ordered[0].y0 + ordered[0].y1) / 2
    ordered[0].line_index = 0

    for word in ordered[1:]:
        center = (word.y0 + word.y1) / 2
        if abs(center - line_center) > tolerance:
            line_index += 1
            line_center = center
        else:
            line_center = (line_center + center) / 2
        word.line_index = line_index

    words.sort(key=lambda w: (w.line_index, -w.x0))


def _bbox_from_poly(points) -> tuple[float, float, float, float]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


def _get(obj, key: str):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
