"""Layer 2 - Render-Quality Search: MultiDPICandidateGenerator.

Single responsibility: call PageRasterizer at every configured DPI,
unconditionally, producing a candidate set. Judges nothing - it does not
score sharpness and does not pick a winner. That is the orchestrator's job,
using SharpnessScorer separately, once both exist.
"""

from __future__ import annotations

import fitz

from app.core.ingestion.page_rasterizer import PageRasterizer, RasterizedPage


class MultiDPICandidateGenerator:
    """Delegates all actual rendering to a PageRasterizer; this module only
    decides which DPIs to ask for."""

    def __init__(self, rasterizer: PageRasterizer | None = None) -> None:
        self._rasterizer = rasterizer or PageRasterizer()

    def generate(self, page: fitz.Page, dpis: tuple[int, ...]) -> list[RasterizedPage]:
        if not dpis:
            raise ValueError("dpis must be a non-empty sequence")
        return [self._rasterizer.rasterize(page, dpi) for dpi in dpis]
