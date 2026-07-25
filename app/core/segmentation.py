"""Line segmentation via horizontal projection profile.

Used to (a) crop individual lines for targeted re-OCR of low-confidence
regions, and (b) give the diacritic-protection logic a baseline band to
reason about. Nastaleeq's diagonal, stepped baseline means a pure
projection-profile approach is cruder than it would be for Naskh/Latin
text, so gaps are merged generously to avoid slicing a line into pieces at
its own descenders/ascenders.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class LineRegion:
    y0: int
    y1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def segment_lines(image: np.ndarray, dpi: int = 600, min_gap_fraction: float = 0.35) -> list[LineRegion]:
    """Returns line bands sorted top-to-bottom. min_gap_fraction (of the
    median line height) controls how large a blank gap must be before two
    text bands are treated as separate lines rather than one line with an
    internal dip (e.g. around a diacritic-heavy word)."""
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_ink = binary.sum(axis=1)
    has_text = row_ink > (binary.shape[1] * 255 * 0.01)  # >1% of the row width is ink

    bands: list[LineRegion] = []
    in_band = False
    start = 0
    for y, flag in enumerate(has_text):
        if flag and not in_band:
            start, in_band = y, True
        elif not flag and in_band:
            bands.append(LineRegion(start, y))
            in_band = False
    if in_band:
        bands.append(LineRegion(start, len(has_text)))

    if not bands:
        return []

    median_height = float(np.median([b.height for b in bands]))
    min_gap = max(3, int(median_height * min_gap_fraction))

    merged: list[LineRegion] = [bands[0]]
    for band in bands[1:]:
        prev = merged[-1]
        if band.y0 - prev.y1 < min_gap:
            merged[-1] = LineRegion(prev.y0, band.y1)
        else:
            merged.append(band)

    return merged


def crop_line(image: np.ndarray, region: LineRegion, padding: int = 4) -> np.ndarray:
    h = image.shape[0]
    y0 = max(0, region.y0 - padding)
    y1 = min(h, region.y1 + padding)
    return image[y0:y1, :]
