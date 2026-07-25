"""Standalone test for Layer 3's GlobalContrastEnhancer (CLAHE) - verifies
it measurably increases contrast on a low-contrast image and doesn't crash
on degenerate input."""

from __future__ import annotations

import numpy as np

from app.core.preprocessing.contrast_enhancer import GlobalContrastEnhancer


def _low_contrast_image(size: int = 300) -> np.ndarray:
    """Content squeezed into a narrow mid-gray band - simulates a faded/
    washed-out scan."""
    rng = np.random.default_rng(0)
    base = rng.integers(110, 145, size=(size, size), dtype=np.uint8)
    return base


def test_clahe_increases_standard_deviation_on_low_contrast_image() -> None:
    image = _low_contrast_image()
    enhanced = GlobalContrastEnhancer().enhance(image)

    assert float(enhanced.std()) > float(image.std())


def test_output_shape_and_dtype_preserved() -> None:
    image = _low_contrast_image()
    enhanced = GlobalContrastEnhancer().enhance(image)

    assert enhanced.shape == image.shape
    assert enhanced.dtype == np.uint8


def test_uniform_image_does_not_crash() -> None:
    uniform = np.full((200, 200), 128, dtype=np.uint8)
    enhanced = GlobalContrastEnhancer().enhance(uniform)
    assert enhanced.shape == uniform.shape


def test_clip_limit_is_configurable() -> None:
    image = _low_contrast_image()
    enhancer = GlobalContrastEnhancer()

    mild = enhancer.enhance(image, clip_limit=1.0)
    strong = enhancer.enhance(image, clip_limit=8.0)

    # Different clip limits must actually produce different output -
    # otherwise the parameter is silently ignored.
    assert not np.array_equal(mild, strong)
