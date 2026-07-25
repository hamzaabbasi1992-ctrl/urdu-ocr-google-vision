"""Benchmark mode: tries a curated grid of DPI/threshold/denoise/sharpen/
contrast combinations against a single page and reports average OCR
confidence per combination.

A full cartesian product of {4 DPIs} x {threshold methods} x {denoise
levels} x {sharpen levels} x {contrast levels} would be hundreds of
full-page OCR runs per page - impractical on this machine (no GPU, so
PaddleOCR/Tesseract are CPU-only). Instead this uses a fixed set of named
preprocessing presets (bundles of denoise/sharpen/contrast/threshold
settings) crossed with the 4 DPIs from the spec, giving a representative
~24-run grid instead of a four-figure one. It only ever runs against pages
the caller explicitly selects, never automatically during batch processing.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import fitz  # PyMuPDF

from app.core.control import JobControl
from app.core.model_manager import ModelManager
from app.core.models import BenchmarkRunResult, BenchmarkVariant, OCRConfig, PreprocessConfig
from app.core.pdf_render import cleanup_document_temp_dir, new_document_temp_dir, render_page
from app.core.preprocess.pipeline import run_pipeline

_LOGGER = logging.getLogger("urdu_ocr.benchmark")

BENCHMARK_DPIS = (300, 600, 900, 1200)

_DENOISE_LEVELS = {"off": 0.0, "light": 0.25, "medium": 0.5, "heavy": 0.8}
_SHARPEN_LEVELS = {"off": 0.0, "light": 0.25, "medium": 0.5, "heavy": 0.8}
_CONTRAST_LEVELS = {"off": 1.0, "light": 1.5, "medium": 2.0, "heavy": 3.0}

# (preset_name, threshold_method|"off", denoise_level, sharpen_level, contrast_level)
_PRESETS: list[tuple[str, str, str, str, str]] = [
    ("baseline", "off", "off", "off", "off"),
    ("light_touch", "off", "light", "light", "light"),
    ("standard", "off", "medium", "medium", "medium"),
    ("heavy_cleanup", "off", "heavy", "heavy", "medium"),
    ("binarized_gaussian", "adaptive_gaussian", "medium", "medium", "medium"),
    ("binarized_otsu", "otsu", "medium", "medium", "medium"),
]


def _variant_config(threshold: str, denoise_lvl: str, sharpen_lvl: str, contrast_lvl: str) -> PreprocessConfig:
    return PreprocessConfig(
        denoise=denoise_lvl != "off",
        denoise_strength=_DENOISE_LEVELS[denoise_lvl],
        sharpen=sharpen_lvl != "off",
        sharpen_amount=_SHARPEN_LEVELS[sharpen_lvl],
        clahe=contrast_lvl != "off",
        clahe_clip_limit=_CONTRAST_LEVELS[contrast_lvl],
        adaptive_threshold=threshold != "off",
        threshold_method=threshold if threshold != "off" else "adaptive_gaussian",
    )


def build_curated_variants(dpis: tuple[int, ...] = BENCHMARK_DPIS) -> list[BenchmarkVariant]:
    """Returns the curated grid of (name, dpi, preprocessing config) variants."""
    variants = []
    for dpi in dpis:
        for name, threshold, denoise_lvl, sharpen_lvl, contrast_lvl in _PRESETS:
            config = _variant_config(threshold, denoise_lvl, sharpen_lvl, contrast_lvl)
            variants.append(BenchmarkVariant(name=f"{name}@{dpi}dpi", dpi=dpi, preprocess=config))
    return variants


def run_benchmark(
    pdf_path,
    page_index: int,
    model_manager: ModelManager,
    ocr_config: OCRConfig,
    variants: list[BenchmarkVariant] | None = None,
    control: JobControl | None = None,
    on_variant_done: Callable[[BenchmarkRunResult], None] | None = None,
) -> list[BenchmarkRunResult]:
    variants = variants if variants is not None else build_curated_variants()
    arbiter = model_manager.get_arbiter(ocr_config)
    results: list[BenchmarkRunResult] = []

    temp_dir = new_document_temp_dir(pdf_path)
    try:
        with fitz.open(pdf_path) as doc:
            for variant in variants:
                if control is not None:
                    control.checkpoint()

                start = time.monotonic()
                render = render_page(doc, page_index, variant.dpi, temp_dir)
                processed, _debug = run_pipeline(render.image, variant.preprocess, variant.dpi)
                words = arbiter.run(processed)
                elapsed = time.monotonic() - start

                avg_confidence = sum(w.confidence for w in words) / len(words) if words else 0.0
                result = BenchmarkRunResult(
                    variant_name=variant.name,
                    dpi=variant.dpi,
                    average_confidence=avg_confidence,
                    word_count=len(words),
                    elapsed_seconds=elapsed,
                    preprocess=variant.preprocess,
                )
                results.append(result)
                _LOGGER.info(
                    "Benchmark %s: avg_confidence=%.3f words=%d (%.1fs)",
                    variant.name, avg_confidence, len(words), elapsed,
                )
                if on_variant_done:
                    on_variant_done(result)
    finally:
        cleanup_document_temp_dir(temp_dir)

    return results


def best_result(results: list[BenchmarkRunResult]) -> BenchmarkRunResult | None:
    if not results:
        return None
    return max(results, key=lambda r: r.average_confidence)
