"""Connected-component analysis used to protect small marks - diacritic dots,
zabar/zer/pesh, and other i'raab - from being erased by denoise/morphology
steps that would otherwise treat any small speck as noise.

This does not try to be a full diacritic classifier. It flags small, compact
components as "protect" candidates; denoise/morphology steps then restore the
original pixels under that mask after processing, so a real noise speck that
happens to be classified conservatively still just costs a little denoise
strength, never a lost dot.
"""

from __future__ import annotations

import cv2
import numpy as np


def label_components(binary_image: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """binary_image: uint8, ink=255 on black background (THRESH_BINARY_INV style)."""
    return cv2.connectedComponentsWithStats(binary_image, connectivity=8)


def protect_mask(
    image: np.ndarray,
    min_area_px: int = 2,
    max_area_px: int = 400,
    min_compactness: float = 0.3,
) -> np.ndarray:
    """Returns a boolean mask (same shape as image) marking pixels that
    belong to small, compact components - the profile of a dot or diacritic
    mark rather than a long joined Nastaleeq stroke or a large noise blob.

    max_area_px should scale with DPI (a dot at 1200 DPI covers more pixels
    than at 600 DPI); callers pick it from the page's chosen render DPI.
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = label_components(binary)

    mask = np.zeros(image.shape, dtype=bool)
    for label_id in range(1, num_labels):  # skip background (0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area_px or area > max_area_px:
            continue
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]
        bbox_area = max(w * h, 1)
        compactness = area / bbox_area  # a filled dot/circle is compact; a thin stroke fragment isn't
        if compactness < min_compactness:
            continue
        mask[labels == label_id] = True

    return mask


def guarded_apply(original: np.ndarray, processed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keeps original pixels wherever mask is True, uses processed elsewhere."""
    return np.where(mask, original, processed).astype(original.dtype)


def dot_area_bounds_for_dpi(dpi: int) -> tuple[int, int]:
    """Rough expected pixel-area range of a diacritic dot at a given render
    DPI, used as protect_mask's min/max_area_px. Calibrated against a
    roughly 1-2mm dot mark."""
    scale = dpi / 600.0
    return max(1, int(2 * scale)), int(400 * scale * scale)
