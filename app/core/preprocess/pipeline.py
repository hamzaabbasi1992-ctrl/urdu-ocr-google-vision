"""Orchestrates every preprocessing step in a fixed, sensible order, honoring
PreprocessConfig's per-step toggles. Each step is independently skippable -
this module just sequences them and records what happened for the GUI's
"show preprocessing image" / debug view.

Also tracks a PageTransform mapping OCR word coordinates (measured on the
final processed image) back to the *canonical* page image - defined as the
image right after orientation-correction and deskew, before crop/resize/
super-resolution. Exports (searchable PDF overlay, JSON bboxes) are built
against that canonical image. Orientation correction (90-degree multiples)
and crop/scale are inverted exactly; deskew's small-angle rotation is not
inverted in the bbox mapping - an accepted approximation, since the
residual positional error from a typically-small skew angle is negligible
for placing a searchable text layer, and inverting a full rotation+crop+
scale composition exactly would add a lot of geometry for little practical
benefit here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from app.core.models import BBox, PreprocessConfig
from app.core.preprocess import contrast, denoise, enhance, geometry, morphology, threshold
from app.core.preprocess.super_resolution import super_resolve

_LOGGER = logging.getLogger("urdu_ocr.preprocess.pipeline")


@dataclass(slots=True)
class PageTransform:
    crop_offset_x: float = 0.0
    crop_offset_y: float = 0.0
    scale: float = 1.0  # processed_coord = (canonical_coord - crop_offset) * scale

    def to_canonical(self, bbox: BBox) -> BBox:
        return BBox(
            x0=bbox.x0 / self.scale + self.crop_offset_x,
            y0=bbox.y0 / self.scale + self.crop_offset_y,
            x1=bbox.x1 / self.scale + self.crop_offset_x,
            y1=bbox.y1 / self.scale + self.crop_offset_y,
        )


@dataclass(slots=True)
class PreprocessDebugInfo:
    applied_steps: list[str] = field(default_factory=list)
    orientation_angle: int = 0
    skew_angle: float = 0.0
    noise_type_detected: str = "unknown"
    faded_region_fraction: float = 0.0
    canonical_image: np.ndarray | None = None  # orientation+deskew corrected, pre-crop/scale
    transform: PageTransform = field(default_factory=PageTransform)


def run_pipeline(
    image: np.ndarray, config: PreprocessConfig, dpi: int
) -> tuple[np.ndarray, PreprocessDebugInfo]:
    debug = PreprocessDebugInfo()
    result = image

    if config.orientation_detect:
        angle = geometry.detect_orientation(result)
        if angle:
            result = geometry.apply_orientation(result, angle)
        debug.orientation_angle = angle
        debug.applied_steps.append("orientation_detect")

    if config.deskew:
        result, skew_angle = geometry.deskew(result)
        debug.skew_angle = skew_angle
        debug.applied_steps.append("deskew")

    debug.canonical_image = result.copy()

    if config.shadow_removal:
        result = denoise.remove_shadow(result)
        debug.applied_steps.append("shadow_removal")

    if config.denoise:
        debug.noise_type_detected = denoise.classify_noise(result)
        result = denoise.denoise(
            result,
            dpi=dpi,
            strength=config.denoise_strength,
            protect_diacritics=config.preserve_diacritics,
            use_classification=config.noise_classification,
        )
        debug.applied_steps.append("denoise")

    if config.auto_crop or config.remove_margins:
        result, (crop_x, crop_y) = geometry.auto_crop_and_remove_margins(result)
        debug.transform.crop_offset_x = float(crop_x)
        debug.transform.crop_offset_y = float(crop_y)
        debug.applied_steps.append("auto_crop_remove_margins")

    if config.super_resolution:
        result = super_resolve(result, scale=config.super_resolution_scale)
        dpi = dpi * config.super_resolution_scale
        debug.transform.scale *= config.super_resolution_scale
        debug.applied_steps.append(f"super_resolution_x{config.super_resolution_scale}")

    if config.resize_if_needed:
        before_shape = result.shape
        result, resize_scale = morphology.resize_if_needed(result, config.min_dimension_px)
        if result.shape != before_shape:
            debug.transform.scale *= resize_scale
            debug.applied_steps.append("resize")

    if config.clahe:
        result = contrast.apply_clahe(result, clip_limit=config.clahe_clip_limit)
        debug.applied_steps.append("clahe")

    if config.adaptive_local_contrast:
        result = contrast.adaptive_local_contrast(result)
        debug.applied_steps.append("adaptive_local_contrast")

    if config.gamma_correction:
        gamma = (
            contrast.estimate_auto_gamma(result) if config.gamma_value == 1.0 else config.gamma_value
        )
        result = contrast.apply_gamma(result, gamma)
        debug.applied_steps.append(f"gamma({gamma:.2f})")

    if config.faded_text_detection:
        faded_mask = enhance.detect_faded_regions(result)
        debug.faded_region_fraction = float(np.mean(faded_mask))
        if debug.faded_region_fraction > 0.001:
            result = enhance.boost_faded_regions(result, faded_mask)
            debug.applied_steps.append("faded_text_boost")

    if config.stroke_enhancement:
        result = enhance.enhance_strokes(result, dpi=dpi, protect_diacritics=config.preserve_diacritics)
        debug.applied_steps.append("stroke_enhancement")

    if config.ink_enhancement:
        result = enhance.enhance_ink(result)
        debug.applied_steps.append("ink_enhancement")

    if config.sharpen:
        result = enhance.sharpen(result, amount=config.sharpen_amount)
        debug.applied_steps.append("sharpen")

    if config.morphological_cleanup:
        result = morphology.morphological_cleanup(result, dpi=dpi, protect_diacritics=config.preserve_diacritics)
        debug.applied_steps.append("morphological_cleanup")

    if config.adaptive_threshold:
        result = threshold.apply_threshold(result, method=config.threshold_method)
        debug.applied_steps.append(f"threshold({config.threshold_method})")

    return result, debug
