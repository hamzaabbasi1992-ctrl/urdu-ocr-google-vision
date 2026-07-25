"""Stroke enhancement, ink enhancement, sharpening, and faded-text detection.

These target the specific failure modes of old/cheaply-printed Islamic book
scans: thin Nastaleeq joins that have partially dropped out, ink that has
faded unevenly across the page, and general softness from the source print.
All strength parameters default mild - a stroke that's over-thickened
merges adjacent letterforms, which is exactly the kind of distortion the
project rules forbid.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.preprocess.components import dot_area_bounds_for_dpi, guarded_apply, protect_mask


def enhance_strokes(image: np.ndarray, dpi: int, protect_diacritics: bool = True) -> np.ndarray:
    """Mildly thickens/darkens ink strokes via a 1-pixel grayscale erosion
    (dark ink = low value, so erode grows dark regions). A single small
    kernel, one iteration - enough to recover a partially-dropped-out
    stroke without fusing separate letterforms."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    processed = cv2.erode(image, kernel, iterations=1)

    if not protect_diacritics:
        return processed
    min_area, max_area = dot_area_bounds_for_dpi(dpi)
    mask = protect_mask(image, min_area_px=min_area, max_area_px=max_area)
    return guarded_apply(image, processed, mask)


def enhance_ink(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Pushes already-dark pixels darker (deepens ink) while leaving
    near-white background mostly alone, via an S-curve LUT. Unlike gamma
    (which reshapes the whole tonal range), this targets the ink band
    specifically."""
    strength = max(0.0, min(1.0, strength))
    x = np.arange(256, dtype=np.float32) / 255.0
    midpoint = 0.6
    steepness = 4 + strength * 8
    curve = 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))
    curve = (curve - curve.min()) / (curve.max() - curve.min())  # renormalize to [0,1]
    table = np.clip(curve * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def sharpen(image: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """Unsharp mask. amount in 0.0-1.0."""
    amount = max(0.0, min(1.0, amount)) * 1.5
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def detect_faded_regions(image: np.ndarray, window: int = 31, min_ink_drop: float = 25.0) -> np.ndarray:
    """Flags pixels that sit in a locally low-contrast area but still show
    some ink signal (local min meaningfully below the local background
    estimate) - i.e. probably-faded text, not blank margin."""
    window = window | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window, window))
    local_min = cv2.erode(image, kernel).astype(np.float32)
    local_max = cv2.dilate(image, kernel).astype(np.float32)
    local_contrast = local_max - local_min

    has_ink_signal = (local_max - local_min) > 5  # not a perfectly flat/blank patch
    is_low_contrast = local_contrast < min_ink_drop * 2
    is_faded = local_min < (local_max - min_ink_drop)  # some real drop below local background

    return has_ink_signal & is_low_contrast & is_faded


def boost_faded_regions(image: np.ndarray, mask: np.ndarray, boost: float = 0.7) -> np.ndarray:
    """Applies extra local contrast stretch only inside the flagged region,
    leaving the rest of the page untouched."""
    from app.core.preprocess.contrast import adaptive_local_contrast

    stretched = adaptive_local_contrast(image, window=21, blend=boost)
    return np.where(mask, stretched, image).astype(image.dtype)
