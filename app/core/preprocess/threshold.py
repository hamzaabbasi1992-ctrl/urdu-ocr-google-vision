"""Adaptive threshold / binarization.

Off by default on the main OCR path (see PreprocessConfig.adaptive_threshold)
because hard binarization risks breaking thin joined Nastaleeq strokes and
erasing small diacritic dots. Kept available as a toggle and as one of the
benchmark grid's dimensions, since some genuinely low-contrast scans do
better with it.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_threshold(image: np.ndarray, method: str = "adaptive_gaussian") -> np.ndarray:
    if method == "otsu":
        _, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result
    if method == "adaptive_mean":
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 15
        )
    # default: adaptive_gaussian
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
