"""CLAHE, adaptive local contrast stretching, and gamma correction."""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(image)


def adaptive_local_contrast(image: np.ndarray, window: int = 31, blend: float = 0.6) -> np.ndarray:
    """Stretches contrast within local windows (via local min/max from
    morphological erode/dilate) so faint ink in a locally low-contrast
    region becomes visible without blowing out already-clear regions.
    `blend` mixes the stretched result with the original to avoid
    over-processing flat background areas."""
    window = window | 1  # must be odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window, window))
    local_min = cv2.erode(image, kernel)
    local_max = cv2.dilate(image, kernel)

    denom = np.maximum(local_max.astype(np.float32) - local_min.astype(np.float32), 1.0)
    stretched = (image.astype(np.float32) - local_min.astype(np.float32)) / denom * 255.0
    stretched = np.clip(stretched, 0, 255)

    blended = blend * stretched + (1 - blend) * image.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def estimate_auto_gamma(image: np.ndarray) -> float:
    mean_brightness = float(image.mean()) / 255.0
    mean_brightness = max(0.05, min(0.95, mean_brightness))
    gamma = np.log(0.5) / np.log(mean_brightness)
    return float(np.clip(gamma, 0.5, 2.5))


def apply_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """gamma == 1.0 is treated by the caller as "use estimate_auto_gamma";
    this function always applies the value it's given literally."""
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)
