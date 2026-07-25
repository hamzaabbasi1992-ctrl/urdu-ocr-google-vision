"""Layer 1 - Ingestion: PageRasterizer.

Single responsibility: render one page (a fitz.Page, e.g. from
PDFLoader.get_page()) to a raster image at one specified DPI. Nothing else -
no quality judgment, no trying multiple DPIs, no deciding anything. DPI
candidate generation and sharpness scoring are Layer 2's job
(MultiDPICandidateGenerator / SharpnessScorer), not this module's.
"""

from __future__ import annotations

from dataclasses import dataclass

import fitz
import numpy as np


@dataclass(slots=True, frozen=True)
class RasterizedPage:
    dpi: int
    image: np.ndarray  # grayscale, uint8, HxW - preserved grayscale, no JPEG


class PageRasterizer:
    """Renders a fitz.Page to a grayscale image at a given DPI."""

    def rasterize(self, page: fitz.Page, dpi: int) -> RasterizedPage:
        if dpi <= 0:
            raise ValueError(f"dpi must be positive, got {dpi}")

        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY, alpha=False)

        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width)
        image = image.copy()  # detach from the pixmap's buffer before it's freed
        return RasterizedPage(dpi=dpi, image=image)
