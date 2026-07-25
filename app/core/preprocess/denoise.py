"""Noise classification, denoising, and shadow removal.

Denoise strength is always applied through components.protect_mask so a
diacritic dot or zabar/zer/pesh mark is restored to its original pixels
after processing, even if the general denoiser would have smoothed it away.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.preprocess.components import dot_area_bounds_for_dpi, guarded_apply, protect_mask

NoiseType = str  # "clean" | "gaussian" | "salt_pepper" | "scanline"


def classify_noise(image: np.ndarray) -> NoiseType:
    median = cv2.medianBlur(image, 3)
    diff = cv2.absdiff(image, median)

    # Salt-and-pepper: a small fraction of pixels differ drastically from their neighborhood
    impulse_fraction = float(np.mean(diff > 40))
    if impulse_fraction > 0.01:
        return "salt_pepper"

    # Scanline: row-mean brightness has high variance relative to the page's overall contrast
    row_means = image.mean(axis=1)
    row_variation = float(np.std(row_means))
    overall_std = float(np.std(image)) or 1.0
    if row_variation > 0.15 * overall_std:
        return "scanline"

    general_noise = float(np.mean(diff))
    if general_noise > 3.0:
        return "gaussian"
    return "clean"


def _flatten_scanlines(image: np.ndarray) -> np.ndarray:
    row_means = image.mean(axis=1, keepdims=True)
    kernel_size = max(3, (image.shape[0] // 50) | 1)  # odd, scales with page height
    smoothed = cv2.GaussianBlur(row_means.astype(np.float32), (1, kernel_size), 0)
    correction = smoothed - row_means
    corrected = image.astype(np.float32) + correction
    return np.clip(corrected, 0, 255).astype(np.uint8)


def denoise(
    image: np.ndarray,
    dpi: int,
    strength: float = 0.5,
    protect_diacritics: bool = True,
    use_classification: bool = True,
) -> np.ndarray:
    """strength in 0.0-1.0. Method is chosen from the detected noise type
    when use_classification is True; otherwise a fixed general-purpose
    (gaussian/NLM) denoise is used regardless of what's actually on the page."""
    noise_type = classify_noise(image) if use_classification else "gaussian"
    strength = max(0.0, min(1.0, strength))

    if noise_type == "clean":
        return image
    elif noise_type == "salt_pepper":
        ksize = 3 if strength < 0.6 else 5
        processed = cv2.medianBlur(image, ksize)
    elif noise_type == "scanline":
        processed = _flatten_scanlines(image)
    else:  # gaussian
        # Bilateral filter, not fastNlMeansDenoising: NLM's block-matching
        # search is dramatically slower per pixel (was the single biggest
        # CPU cost in the whole pipeline, taking minutes even after capping
        # the working resolution) and its extra quality over a bilateral
        # filter isn't worth that on a CPU-only machine. Bilateral still
        # preserves edges/strokes while smoothing background noise.
        sigma = 15 + strength * 45
        processed = cv2.bilateralFilter(image, d=9, sigmaColor=sigma, sigmaSpace=sigma)

    if not protect_diacritics:
        return processed

    min_area, max_area = dot_area_bounds_for_dpi(dpi)
    mask = protect_mask(image, min_area_px=min_area, max_area_px=max_area)
    return guarded_apply(image, processed, mask)


def remove_shadow(image: np.ndarray) -> np.ndarray:
    """Estimates uneven-illumination/shadow background via a large-kernel
    dilate+median-blur pass, then normalizes it out. Standard technique for
    scans with a shadow gradient (e.g. a book photographed instead of a
    flatbed scan)."""
    dilated = cv2.dilate(image, np.ones((7, 7), np.uint8))
    background = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(image, background)
    normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)
