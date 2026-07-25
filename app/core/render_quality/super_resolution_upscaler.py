"""Layer 2 - Render-Quality Search: SuperResolutionUpscaler.

Single responsibility: given one image, return an upscaled version. Does
not decide *whether* super-resolution should run for a given page - that
is the orchestrator's job (Layer 11), likely triggered by SharpnessScorer's
output on a page that is still soft after DPI escalation.

FSRCNN (via OpenCV's dnn_superres, part of opencv-contrib-python - already
a required dependency, not a new one) is used rather than a GAN-based
model: this machine has no GPU, and a GAN model's per-page CPU cost would
be impractical. This step's justification (per PROJECT_SPEC.md Section 3,
which requires measured evidence, not a vendor's self-reported claim) is an
independent research benchmark - Sarim et al., "From Press to Pixels:
Evolving Urdu Text Recognition" (2025) - which found super-resolution
improved downstream Urdu OCR accuracy by 25-70%.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

_LOGGER = logging.getLogger("urdu_ocr.render_quality.super_resolution")

_MODEL_URLS = {
    2: "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x2.pb",
    3: "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x3.pb",
    4: "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x4.pb",
}


class SuperResolutionUpscaler:
    """Upscales an image by a fixed integer factor via FSRCNN. Falls back
    to a plain Lanczos resize (logged, not silent) if the model weights
    aren't available locally and can't be fetched - offline-safe."""

    def __init__(self, model_dir: Path, scale: int = 2) -> None:
        if scale not in _MODEL_URLS:
            raise ValueError(f"Unsupported scale {scale}; supported: {sorted(_MODEL_URLS)}")
        self._scale = scale
        self._model_path = Path(model_dir) / f"FSRCNN_x{scale}.pb"
        self._engine = None  # lazy-loaded

    def upscale(self, image: np.ndarray) -> np.ndarray:
        engine = self._get_engine()
        if engine is None:
            h, w = image.shape[:2]
            return cv2.resize(
                image, (w * self._scale, h * self._scale), interpolation=cv2.INTER_LANCZOS4
            )

        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        upscaled_bgr = engine.upsample(bgr)
        return cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2GRAY)

    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        if not self._model_path.exists() and not self._download_model():
            return None

        engine = cv2.dnn_superres.DnnSuperResImpl_create()
        engine.readModel(str(self._model_path))
        engine.setModel("fsrcnn", self._scale)
        self._engine = engine
        return engine

    def _download_model(self) -> bool:
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _LOGGER.info("Downloading FSRCNN x%d model (one-time)...", self._scale)
            urllib.request.urlretrieve(_MODEL_URLS[self._scale], str(self._model_path))
            return True
        except Exception as exc:  # noqa: BLE001 - offline/no-network must degrade, not crash
            _LOGGER.warning(
                "Could not download FSRCNN model (%s); falling back to plain resize.", exc
            )
            return False
