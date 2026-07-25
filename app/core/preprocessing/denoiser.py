"""Layer 3 - Preprocessing Transforms: Denoiser.

Single responsibility: denoise only. Pure `image -> image` function - does
not decide whether to run, does not know about any other preprocessing
step.

Minimal version (per PROJECT_SPEC.md Section 4): a single fixed strategy -
no NoiseClassifier, no per-noise-type dispatch. Uses a bilateral filter,
not `cv2.fastNlMeansDenoising` - ported from the old
`app/core/preprocess/denoise.py`'s "gaussian" branch, where NLM was found
to be the single biggest CPU-time cost in the entire old pipeline (minutes
per page) with no measured accuracy benefit over a bilateral filter to
justify it. Re-deriving this from scratch would risk reintroducing that
exact mistake.
"""

from __future__ import annotations

import cv2
import numpy as np


class Denoiser:
    def denoise(self, image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        strength = max(0.0, min(1.0, strength))
        sigma = 15 + strength * 45
        return cv2.bilateralFilter(image, d=9, sigmaColor=sigma, sigmaSpace=sigma)
