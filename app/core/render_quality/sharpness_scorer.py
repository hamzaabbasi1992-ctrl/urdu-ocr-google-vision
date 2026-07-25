"""Layer 2 - Render-Quality Search: SharpnessScorer.

Single responsibility: given one image, return a sharpness metric. Pure
function - does not render, does not compare candidates, does not decide
anything about DPI. That is MultiDPICandidateGenerator's job.
"""

from __future__ import annotations

import cv2
import numpy as np

# Normalize before scoring so a bigger image isn't scored "sharper" just
# because it has more pixels - comparisons across different rendered DPIs
# would otherwise be meaningless.
_COMPARISON_WIDTH = 2000


class SharpnessScorer:
    """Laplacian-variance sharpness scorer, normalized to a fixed width."""

    def score(self, image: np.ndarray) -> float:
        h, w = image.shape[:2]
        if w > _COMPARISON_WIDTH:
            scale = _COMPARISON_WIDTH / w
            image = cv2.resize(
                image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        return float(cv2.Laplacian(image, cv2.CV_64F).var())
