"""Morphological cleanup (dot-safe) and resizing."""

from __future__ import annotations

import cv2
import numpy as np

from app.core.preprocess.components import dot_area_bounds_for_dpi, guarded_apply, protect_mask


def morphological_cleanup(image: np.ndarray, dpi: int, protect_diacritics: bool = True) -> np.ndarray:
    """Removes stray single/few-pixel speckle noise via a tiny opening.
    Kernel is deliberately 2x2 (not 3x3+) since Nastaleeq strokes and dots
    are thin - anything larger risks erasing real marks even with the
    protect mask guarding classified dots."""
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Re-apply as grayscale: wherever opening removed ink, restore background there
    removed = cv2.bitwise_and(binary, cv2.bitwise_not(opened))
    processed = image.copy()
    processed[removed > 0] = 255

    if not protect_diacritics:
        return processed
    min_area, max_area = dot_area_bounds_for_dpi(dpi)
    mask = protect_mask(image, min_area_px=min_area, max_area_px=max_area)
    return guarded_apply(image, processed, mask)


def resize_if_needed(image: np.ndarray, min_dimension_px: int = 1600) -> tuple[np.ndarray, float]:
    """Returns the (possibly resized) image and the scale factor applied
    (1.0 if unchanged), so callers can map coordinates back to the
    pre-resize image."""
    h, w = image.shape[:2]
    smaller_side = min(h, w)
    if smaller_side >= min_dimension_px:
        return image, 1.0
    scale = min_dimension_px / smaller_side
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4), scale
