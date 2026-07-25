"""Layer 3 - Preprocessing Transforms: Deskewer.

Single responsibility: detect and correct fine skew angle only (a scan laid
slightly crooked). Pure `image -> image` function - does not decide whether
to run, does not know about any other preprocessing step.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_MAX_CORRECTION_DEGREES = 15.0  # beyond this, a "skew" reading is more likely noise
_MIN_CORRECTION_DEGREES = 0.05  # below this, correction isn't worth the interpolation cost


@dataclass(slots=True, frozen=True)
class DeskewResult:
    image: np.ndarray
    angle_degrees: float


class Deskewer:
    def deskew(self, image: np.ndarray) -> DeskewResult:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(binary)
        if coords is None or len(coords) < 50:
            return DeskewResult(image=image, angle_degrees=0.0)

        # cv2.minAreaRect's angle convention has changed across OpenCV
        # versions (some return the small skew angle directly in [-45, 45),
        # others return an angle near 90 measured from a different
        # reference edge for wide, mostly-horizontal content). Reducing
        # modulo 90 and folding into (-45, 45] gives the smallest-magnitude
        # equivalent angle regardless of which convention this build uses -
        # verified against this exact installed OpenCV build, which returns
        # ~84 degrees (not ~-6) for a real 6-degree rotation.
        angle = cv2.minAreaRect(coords)[-1] % 90
        if angle > 45:
            angle -= 90

        if abs(angle) > _MAX_CORRECTION_DEGREES or abs(angle) < _MIN_CORRECTION_DEGREES:
            return DeskewResult(image=image, angle_degrees=0.0)

        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return DeskewResult(image=rotated, angle_degrees=float(angle))
