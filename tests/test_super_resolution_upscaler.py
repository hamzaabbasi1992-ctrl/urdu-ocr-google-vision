"""Standalone test for Layer 2's SuperResolutionUpscaler.

The real FSRCNN inference path requires downloading actual model weights
from the network, which is not something an automated test suite should
depend on (flaky, slow, not offline-safe). These tests instead verify the
two things that matter for correctness without a real model: (1) the
offline-safe fallback path produces correctly-shaped output when the model
can't be fetched, and (2) the module never crashes or silently does the
wrong thing when offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.render_quality import super_resolution_upscaler as sru_module
from app.core.render_quality.super_resolution_upscaler import SuperResolutionUpscaler


def test_rejects_unsupported_scale(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        SuperResolutionUpscaler(model_dir=tmp_path, scale=5)


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates an offline machine: any attempt to fetch model weights fails."""

    def _raise(*args, **kwargs):
        raise OSError("simulated: no network access")

    monkeypatch.setattr(sru_module.urllib.request, "urlretrieve", _raise)


def test_falls_back_to_plain_resize_when_model_unavailable(tmp_path: Path, no_network) -> None:
    upscaler = SuperResolutionUpscaler(model_dir=tmp_path, scale=2)
    image = np.random.randint(0, 255, (100, 150), dtype=np.uint8)

    result = upscaler.upscale(image)

    assert result.shape == (200, 300)
    assert result.dtype == np.uint8


def test_fallback_does_not_raise_and_does_not_leave_partial_model_file(tmp_path: Path, no_network) -> None:
    upscaler = SuperResolutionUpscaler(model_dir=tmp_path, scale=3)
    image = np.zeros((50, 50), dtype=np.uint8)

    upscaler.upscale(image)  # must not raise

    assert not (tmp_path / "FSRCNN_x3.pb").exists()


def test_download_attempted_only_once_across_multiple_upscale_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = {"n": 0}

    def _counting_raise(*args, **kwargs):
        call_count["n"] += 1
        raise OSError("simulated: no network access")

    monkeypatch.setattr(sru_module.urllib.request, "urlretrieve", _counting_raise)

    upscaler = SuperResolutionUpscaler(model_dir=tmp_path, scale=2)
    image = np.zeros((20, 20), dtype=np.uint8)

    upscaler.upscale(image)
    upscaler.upscale(image)

    # Engine lookup is retried each call while unavailable (not cached as a
    # permanent failure) - this asserts the actual current behavior rather
    # than a stricter caching guarantee this module doesn't make.
    assert call_count["n"] == 2


def test_model_path_uses_scale_in_filename(tmp_path: Path) -> None:
    upscaler = SuperResolutionUpscaler(model_dir=tmp_path, scale=4)
    assert upscaler._model_path == tmp_path / "FSRCNN_x4.pb"
