"""Layer 6 - Recognition Engines: shared recognition output type.

`RecognizedWord` and `assign_reading_order` are engine-agnostic - every
recognition engine's `recognize()` returns `RecognizedWord`s and orders them
via `assign_reading_order`, so this lives outside any single engine module.
Originally part of `paddle_ocr_engine.py` (folded in per the
codebase-minimization directive, back when PaddleOCREngine was the only
engine); split out when PaddleOCREngine was removed, since GoogleVisionEngine
and TextExporter still depend on both.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RecognizedWord:
    text: str
    confidence: float
    x0: float
    y0: float
    x1: float
    y1: float
    line_index: int = 0


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
