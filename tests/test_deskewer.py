"""Standalone test for Layer 3's Deskewer - verifies it detects and
corrects a known skew angle, leaves already-straight/blank images alone,
and ignores implausibly large "skew" readings."""

from __future__ import annotations

import cv2
import numpy as np

from app.core.preprocessing.deskewer import Deskewer


def _text_like_image(angle_deg: float = 0.0, size: int = 400) -> np.ndarray:
    """Several horizontal lines simulate text lines on a page; rotating the
    whole image simulates a crookedly-scanned page."""
    image = np.full((size, size), 255, dtype=np.uint8)
    for y in range(60, size - 60, 30):
        cv2.line(image, (50, y), (size - 50, y), color=0, thickness=4)

    if angle_deg != 0:
        center = (size / 2, size / 2)
        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        image = cv2.warpAffine(image, matrix, (size, size), borderValue=255)
    return image


def test_deskew_detects_significant_angle() -> None:
    rotated = _text_like_image(angle_deg=6.0)
    result = Deskewer().deskew(rotated)
    assert abs(result.angle_degrees) > 2.0  # a real, non-trivial correction was applied


def test_deskew_correction_converges_towards_straight() -> None:
    """Sign-convention-agnostic check: re-running deskew on the corrected
    image should find it much straighter than the original."""
    rotated = _text_like_image(angle_deg=6.0)
    deskewer = Deskewer()

    first = deskewer.deskew(rotated)
    second = deskewer.deskew(first.image)

    assert abs(second.angle_degrees) < abs(first.angle_degrees)


def test_already_straight_image_is_left_unchanged() -> None:
    straight = _text_like_image(angle_deg=0.0)
    result = Deskewer().deskew(straight)
    assert result.angle_degrees == 0.0
    assert np.array_equal(result.image, straight)


def test_blank_image_is_left_unchanged() -> None:
    blank = np.full((400, 400), 255, dtype=np.uint8)
    result = Deskewer().deskew(blank)
    assert result.angle_degrees == 0.0
    assert np.array_equal(result.image, blank)


def test_implausibly_large_angle_is_ignored() -> None:
    """A near-90-degree 'skew' on a page is almost certainly a bad
    minAreaRect reading (e.g. from sparse content), not a real skew - the
    module should refuse to apply it rather than rotating the page sideways."""
    extreme = _text_like_image(angle_deg=40.0)
    result = Deskewer().deskew(extreme)
    assert result.angle_degrees == 0.0
    assert np.array_equal(result.image, extreme)
