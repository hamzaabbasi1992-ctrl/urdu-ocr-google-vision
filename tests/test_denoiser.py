"""Standalone test for Layer 3's Denoiser - verifies it measurably reduces
noise while behaving sanely on edge cases."""

from __future__ import annotations

import numpy as np

from app.core.preprocessing.denoiser import Denoiser


def _noisy_version_of(clean: np.ndarray, seed: int = 0, sigma: float = 20.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = clean.astype(np.float32) + rng.normal(0, sigma, size=clean.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _clean_image(size: int = 200) -> np.ndarray:
    image = np.full((size, size), 255, dtype=np.uint8)
    image[50:150, 50:150] = 0  # a solid block, like a large ink region
    return image


def test_denoise_reduces_distance_to_clean_original() -> None:
    clean = _clean_image()
    noisy = _noisy_version_of(clean)
    denoised = Denoiser().denoise(noisy)

    noisy_error = float(np.mean((noisy.astype(np.float32) - clean.astype(np.float32)) ** 2))
    denoised_error = float(np.mean((denoised.astype(np.float32) - clean.astype(np.float32)) ** 2))

    assert denoised_error < noisy_error


def test_output_shape_and_dtype_preserved() -> None:
    noisy = _noisy_version_of(_clean_image())
    denoised = Denoiser().denoise(noisy)
    assert denoised.shape == noisy.shape
    assert denoised.dtype == np.uint8


def test_strength_parameter_changes_output() -> None:
    noisy = _noisy_version_of(_clean_image())
    denoiser = Denoiser()

    mild = denoiser.denoise(noisy, strength=0.0)
    strong = denoiser.denoise(noisy, strength=1.0)

    assert not np.array_equal(mild, strong)


def test_strength_out_of_range_is_clamped_not_rejected() -> None:
    noisy = _noisy_version_of(_clean_image())
    denoiser = Denoiser()

    # Must not raise - out-of-range strength is clamped to [0, 1], not an error.
    denoiser.denoise(noisy, strength=-5.0)
    denoiser.denoise(noisy, strength=99.0)
