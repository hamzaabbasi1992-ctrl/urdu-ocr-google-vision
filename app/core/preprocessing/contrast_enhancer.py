"""Layer 3 - Preprocessing Transforms: GlobalContrastEnhancer (CLAHE).

Single responsibility: global contrast enhancement only. Pure
`image -> image` function - does not decide whether to run, does not know
about any other preprocessing step.

Logic ported unchanged from the old `app/core/preprocess/contrast.py`
(`apply_clahe`) rather than re-derived - per PROJECT_SPEC.md Section 4/7,
rewriting a correct, parameter-tuned CLAHE call would cost time for zero
measurable accuracy difference.
"""

from __future__ import annotations

import cv2
import numpy as np


class GlobalContrastEnhancer:
    def enhance(self, image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(image)
