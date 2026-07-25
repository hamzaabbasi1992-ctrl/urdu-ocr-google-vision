"""Step 1-2: open a PDF and render pages to high-resolution grayscale PNGs.

Adaptive DPI escalation: render at the base DPI (600 by spec) and measure
sharpness with Laplacian variance. If the page looks soft (a common symptom
of a low-quality scan re-photographed/re-scanned at low source resolution),
re-render at progressively higher DPI (900, then 1200) and keep whichever
candidate is sharpest. Sharpness is compared on a common resized copy so the
comparison isn't just measuring "more pixels = higher raw variance".

All candidate PNGs are written under a per-document temp folder; only the
chosen page's outputs persist past `cleanup_page_render` - everything else
is scratch and is deleted once a page's result is committed.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np

from app.core.models import OCRConfig
from app.core.paths import temp_dir

_LOGGER = logging.getLogger("urdu_ocr.pdf_render")

_SHARPNESS_COMPARISON_WIDTH = 2000  # normalize before comparing Laplacian variance across DPIs


@dataclass(slots=True)
class PageRender:
    dpi: int
    image: np.ndarray  # grayscale, uint8, HxW
    png_path: Path


@dataclass(slots=True)
class AdaptiveRenderResult:
    chosen: PageRender
    sharpness_score: float
    tried_dpis: list[int]
    discarded: list[PageRender]


def open_pdf(pdf_path: Path) -> fitz.Document:
    return fitz.open(pdf_path)


def page_count(pdf_path: Path) -> int:
    with fitz.open(pdf_path) as doc:
        return doc.page_count


def render_page(doc: fitz.Document, page_index: int, dpi: int, out_dir: Path) -> PageRender:
    """Render one page at the given DPI to a grayscale PNG (no JPEG compression)."""
    page = doc[page_index]
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY, alpha=False)

    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width)
    image = image.copy()  # detach from the pixmap's underlying buffer before it's freed

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"page{page_index + 1:04d}_dpi{dpi}.png"
    cv2.imwrite(str(png_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    return PageRender(dpi=dpi, image=image, png_path=png_path)


def laplacian_sharpness(image: np.ndarray) -> float:
    """Laplacian variance, computed on a width-normalized copy so scores are
    comparable across images rendered at different DPI/resolution."""
    h, w = image.shape[:2]
    if w > _SHARPNESS_COMPARISON_WIDTH:
        scale = _SHARPNESS_COMPARISON_WIDTH / w
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def render_page_adaptive(
    doc: fitz.Document, page_index: int, config: OCRConfig, out_dir: Path
) -> AdaptiveRenderResult:
    """Render at config.base_dpi; if too soft and adaptive_dpi is enabled,
    escalate through config.dpi_escalation_steps and keep the sharpest."""
    steps = [d for d in config.dpi_escalation_steps if d >= config.base_dpi] or [config.base_dpi]
    if config.base_dpi not in steps:
        steps = [config.base_dpi, *steps]

    tried: list[PageRender] = []
    best: PageRender | None = None
    best_score = -1.0

    for dpi in steps:
        render = render_page(doc, page_index, dpi, out_dir)
        score = laplacian_sharpness(render.image)
        tried.append(render)
        _LOGGER.debug("Page %d at %d DPI: sharpness=%.1f", page_index + 1, dpi, score)

        if score > best_score:
            best, best_score = render, score

        good_enough = score >= config.sharpness_threshold
        if not config.adaptive_dpi or good_enough:
            break

    assert best is not None
    discarded = [r for r in tried if r is not best]
    best = _cap_working_dimension(best, config.max_working_dimension_px)
    return AdaptiveRenderResult(
        chosen=best, sharpness_score=best_score, tried_dpis=[r.dpi for r in tried], discarded=discarded
    )


def _cap_working_dimension(render: PageRender, max_dimension_px: int) -> PageRender:
    """Downscales the winning render if it exceeds the working-resolution
    cap (see OCRConfig.max_working_dimension_px) - DPI escalation is about
    recovering detail from a blurry scan, not about how many pixels the CV
    pipeline and OCR engines then have to churn through. `dpi` is adjusted
    proportionally so downstream point<->pixel conversions (searchable-PDF
    export) stay consistent with the actual working image."""
    h, w = render.image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_dimension_px:
        return render

    scale = max_dimension_px / long_side
    resized = cv2.resize(render.image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    effective_dpi = max(1, round(render.dpi * scale))
    _LOGGER.debug(
        "Capping working resolution: %dx%d @ %d DPI -> %dx%d @ %d DPI",
        w, h, render.dpi, resized.shape[1], resized.shape[0], effective_dpi,
    )
    return PageRender(dpi=effective_dpi, image=resized, png_path=render.png_path)


def cleanup_page_render(result: AdaptiveRenderResult, keep_chosen: bool = True) -> None:
    """Delete scratch PNGs. If keep_chosen is False, the chosen render's file
    is deleted too (use when the caller has already persisted its own copy)."""
    for render in result.discarded:
        render.png_path.unlink(missing_ok=True)
    if not keep_chosen:
        result.chosen.png_path.unlink(missing_ok=True)


def new_document_temp_dir(pdf_path: Path) -> Path:
    path = temp_dir() / f"render_{pdf_path.stem}_{id(pdf_path)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_document_temp_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
