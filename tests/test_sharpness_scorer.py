"""Standalone test for Layer 2's SharpnessScorer - verifies it correctly
distinguishes sharp from blurred content and that its width-normalization
prevents "more pixels" from being scored as "sharper"."""

from __future__ import annotations

import cv2
import numpy as np

from app.core.render_quality.sharpness_scorer import SharpnessScorer


def _checkerboard(size: int = 800, square: int = 10) -> np.ndarray:
    """High-frequency content - the kind of edge detail real text produces."""
    image = np.zeros((size, size), dtype=np.uint8)
    for y in range(0, size, square * 2):
        for x in range(0, size, square * 2):
            image[y : y + square, x : x + square] = 255
            image[y + square : y + 2 * square, x + square : x + 2 * square] = 255
    return image


def test_sharp_image_scores_higher_than_blurred() -> None:
    sharp = _checkerboard()
    blurred = cv2.GaussianBlur(sharp, (15, 15), 0)

    scorer = SharpnessScorer()
    sharp_score = scorer.score(sharp)
    blurred_score = scorer.score(blurred)

    assert sharp_score > blurred_score * 5  # blur should crush high-frequency content


def test_blank_image_scores_near_zero() -> None:
    blank = np.full((800, 800), 255, dtype=np.uint8)
    scorer = SharpnessScorer()
    assert scorer.score(blank) < 1.0


def test_score_is_pure_no_side_effects() -> None:
    image = _checkerboard()
    scorer = SharpnessScorer()
    first = scorer.score(image)
    second = scorer.score(image)
    assert first == second


def test_normalization_prevents_pure_upscale_from_inflating_score() -> None:
    """A low-res image naively upscaled has no new real detail - its score
    after normalization should be in the same ballpark as the original, not
    proportional to its new (larger) pixel count."""
    small_sharp = _checkerboard(size=400, square=5)
    upscaled = cv2.resize(small_sharp, (3000, 3000), interpolation=cv2.INTER_NEAREST)

    scorer = SharpnessScorer()
    small_score = scorer.score(small_sharp)
    upscaled_score = scorer.score(upscaled)

    # Same ballpark (within an order of magnitude), not wildly inflated by
    # sheer pixel count the way an un-normalized Laplacian variance would be.
    assert upscaled_score < small_score * 10
