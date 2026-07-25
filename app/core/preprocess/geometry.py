"""Orientation detection, deskew, automatic page cropping, and margin removal.

All functions take and return a single-channel (grayscale) uint8 image and
are individually toggleable from PreprocessConfig - each is a pure function
so the pipeline can skip any of them without touching the others.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

_LOGGER = logging.getLogger("urdu_ocr.preprocess.geometry")

_MAX_DESKEW_CORRECTION_DEGREES = 15.0  # beyond this, a "skew" reading is more likely noise


def detect_orientation(image: np.ndarray) -> int:
    """Returns the page rotation needed in degrees (0/90/180/270), via
    Tesseract's orientation-and-script-detection. Falls back to 0 (no
    rotation) if OSD can't find enough text to be confident - safer than
    guessing on a sparse/noisy page."""
    try:
        import pytesseract

        osd = pytesseract.image_to_osd(image)
        for line in osd.splitlines():
            if line.startswith("Rotate:"):
                return int(line.split(":")[1].strip())
    except Exception as exc:  # noqa: BLE001 - OSD failing must not abort the pipeline
        _LOGGER.debug("Orientation detection skipped: %s", exc)
    return 0


def apply_orientation(image: np.ndarray, rotate_degrees: int) -> np.ndarray:
    if rotate_degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotate_degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotate_degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Corrects small skew (a scan laid slightly crooked). Returns the
    corrected image and the applied angle in degrees."""
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 50:
        return image, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns an angle in [-90, 0); normalize to a signed small-angle correction
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > _MAX_DESKEW_CORRECTION_DEGREES:
        _LOGGER.debug("Skipping deskew: measured angle %.2f exceeds sane bound", angle)
        return image, 0.0
    if abs(angle) < 0.05:
        return image, 0.0

    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, float(angle)


def auto_crop_and_remove_margins(
    image: np.ndarray, padding_fraction: float = 0.01
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crops to the content bounding box (drops black scan borders and blank
    margins), keeping a small padding so nothing touching the edge is
    clipped. Returns the cropped image and the (x, y) offset that was cut
    from the top-left, so callers can map coordinates back to the
    pre-crop image (needed to place OCR word boxes on the original scan
    for the searchable-PDF/JSON exports)."""
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Close small gaps so a page of many separate glyphs merges into one content blob
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, (0, 0)

    h, w = image.shape[:2]
    x0, y0, x1, y1 = w, h, 0, 0
    page_area = h * w
    for contour in contours:
        if cv2.contourArea(contour) < page_area * 0.0005:
            continue  # ignore specks - not real content blocks
        x, y, cw, ch = cv2.boundingRect(contour)
        x0, y0 = min(x0, x), min(y0, y)
        x1, y1 = max(x1, x + cw), max(y1, y + ch)

    if x1 <= x0 or y1 <= y0:
        return image, (0, 0)

    pad_x, pad_y = int(w * padding_fraction), int(h * padding_fraction)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)

    return image[y0:y1, x0:x1], (x0, y0)
