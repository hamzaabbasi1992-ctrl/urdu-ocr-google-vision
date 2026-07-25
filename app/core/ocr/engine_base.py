"""Shared contract for OCR engines, and the small utility every engine uses
to turn its raw per-region results into line-grouped OCRWord objects.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from app.core.models import BBox, OCRWord


class OCREngine(Protocol):
    name: str

    def run(self, image: np.ndarray) -> list[OCRWord]: ...


def assign_line_indices(words: list[OCRWord]) -> None:
    """Groups words into lines by vertical center, tolerant of Nastaleeq's
    diagonal/stepped baseline (a generous tolerance band relative to the
    median word height, rather than a strict row match). Mutates in place."""
    if not words:
        return

    heights = [w.bbox.y1 - w.bbox.y0 for w in words]
    median_height = sorted(heights)[len(heights) // 2] or 1.0
    tolerance = median_height * 0.6

    ordered = sorted(words, key=lambda w: (w.bbox.y0 + w.bbox.y1) / 2)
    line_index = 0
    line_center = (ordered[0].bbox.y0 + ordered[0].bbox.y1) / 2
    ordered[0].line_index = 0

    for word in ordered[1:]:
        center = (word.bbox.y0 + word.bbox.y1) / 2
        if abs(center - line_center) > tolerance:
            line_index += 1
            line_center = center
        else:
            # running average keeps the band from drifting over a long line
            line_center = (line_center + center) / 2
        word.line_index = line_index

    # Within each line, order right-to-left (Urdu/Nastaleeq reading order)
    words.sort(key=lambda w: (w.line_index, -w.bbox.x0))


def bbox_iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = (a.x1 - a.x0) * (a.y1 - a.y0)
    area_b = (b.x1 - b.x0) * (b.y1 - b.y0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def polygon_to_bbox(points: "list[tuple[float, float]] | np.ndarray") -> BBox:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return BBox(
        x0=float(pts[:, 0].min()),
        y0=float(pts[:, 1].min()),
        x1=float(pts[:, 0].max()),
        y1=float(pts[:, 1].max()),
    )
