"""Multi-scale OCR: runs the arbiter across every rendered DPI variant of a
page (not a synthetic scale pyramid - the adaptive-DPI renders already are
the scales, see pdf_render.py) and merges results by confidence.

Opt-in (OCRConfig.multiscale_ocr) since it multiplies OCR cost by the
number of DPI variants tried - meaningful accuracy gain on marginal pages,
but not worth paying for on every page of a large batch by default.
"""

from __future__ import annotations

from app.core.models import BBox, OCRWord
from app.core.ocr.arbiter import Arbiter
from app.core.ocr.engine_base import bbox_iou


def run_multiscale(images_by_dpi: dict[int, "object"], reference_dpi: int, arbiter: Arbiter) -> list[OCRWord]:
    per_scale_words: dict[int, list[OCRWord]] = {}
    for dpi, image in images_by_dpi.items():
        words = arbiter.run(image)
        factor = reference_dpi / dpi
        for word in words:
            b = word.bbox
            word.bbox = BBox(b.x0 * factor, b.y0 * factor, b.x1 * factor, b.y1 * factor)
        per_scale_words[dpi] = words

    ordered_dpis = sorted(per_scale_words, key=lambda d: 0 if d == reference_dpi else 1)
    merged: list[OCRWord] = list(per_scale_words.get(ordered_dpis[0], []))

    for dpi in ordered_dpis[1:]:
        for word in per_scale_words[dpi]:
            best, best_iou = None, 0.2
            for existing in merged:
                iou = bbox_iou(word.bbox, existing.bbox)
                if iou > best_iou:
                    best, best_iou = existing, iou
            if best is None:
                merged.append(word)
            elif word.confidence > best.confidence:
                merged[merged.index(best)] = word

    return merged
