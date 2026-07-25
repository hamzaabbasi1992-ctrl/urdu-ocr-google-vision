"""Optional super-resolution upscaling before OCR, via OpenCV's dnn_superres
with FSRCNN. FSRCNN is used deliberately instead of a GAN-based model
(Real-ESRGAN etc.) - this machine has no GPU, and a GAN model's per-page
CPU cost would make batch processing impractical. FSRCNN is small and fast
enough to run per-page on CPU while still recovering detail a plain resize
can't.

Off by default (PreprocessConfig.super_resolution) given the CPU cost; the
GUI can enable it per-document, or the pipeline can auto-trigger it when
DPI escalation still leaves a page below the sharpness threshold.
"""

from __future__ import annotations

import logging
import urllib.request

import cv2
import numpy as np

from app.core.paths import model_cache_dir

_LOGGER = logging.getLogger("urdu_ocr.preprocess.super_resolution")

_MODEL_URLS = {
    2: "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x2.pb",
    3: "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x3.pb",
    4: "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x4.pb",
}

_engine_cache: dict[int, cv2.dnn_superres.DnnSuperResImpl] = {}


def _model_path(scale: int) -> "object":
    return model_cache_dir() / "superres" / f"FSRCNN_x{scale}.pb"


def download_fsrcnn_model(scale: int) -> bool:
    """One-time fetch of the FSRCNN weights. Returns False (without raising)
    if there's no network available - callers fall back to a plain resize."""
    path = _model_path(scale)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return True
    try:
        _LOGGER.info("Downloading FSRCNN x%d super-resolution model (one-time)...", scale)
        urllib.request.urlretrieve(_MODEL_URLS[scale], str(path))
        return True
    except Exception as exc:  # noqa: BLE001 - offline/no-network must degrade, not crash
        _LOGGER.warning("Could not download FSRCNN model (%s); super-resolution will fall back to resize.", exc)
        return False


def _get_engine(scale: int) -> cv2.dnn_superres.DnnSuperResImpl | None:
    if scale in _engine_cache:
        return _engine_cache[scale]

    path = _model_path(scale)
    if not path.exists() and not download_fsrcnn_model(scale):
        return None

    engine = cv2.dnn_superres.DnnSuperResImpl_create()
    engine.readModel(str(path))
    engine.setModel("fsrcnn", scale)
    _engine_cache[scale] = engine
    return engine


def super_resolve(image: np.ndarray, scale: int = 2) -> np.ndarray:
    """Upscales a grayscale image by `scale`. Falls back to Lanczos resize
    if the FSRCNN weights aren't available locally and can't be fetched."""
    engine = _get_engine(scale)
    if engine is None:
        h, w = image.shape[:2]
        return cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    # dnn_superres expects a 3-channel image
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    upscaled_bgr = engine.upsample(bgr)
    return cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2GRAY)
