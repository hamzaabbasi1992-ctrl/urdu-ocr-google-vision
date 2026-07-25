"""Standalone test for Layer 1's PageRasterizer - verifies its single
responsibility (render one page at one DPI to a grayscale image) in
isolation. Uses PDFLoader only to obtain a real fitz.Page to render; it
does not test PDFLoader's own behavior (see test_pdf_loader.py)."""

from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
import pytest

from app.core.ingestion.page_rasterizer import PageRasterizer
from app.core.ingestion.pdf_loader import PDFLoader


@pytest.fixture
def page_with_content(tmp_path: Path):
    """A single US-Letter (612x792pt) page with a drawn black rectangle,
    so the rasterized output isn't just a blank/uniform image."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_rect(fitz.Rect(100, 100, 400, 200), color=(0, 0, 0), fill=(0, 0, 0))
    path = tmp_path / "content.pdf"
    doc.save(path)
    doc.close()

    with PDFLoader(path) as loader:
        yield loader.get_page(0)


def test_rasterize_produces_expected_pixel_dimensions(page_with_content) -> None:
    rasterizer = PageRasterizer()
    result = rasterizer.rasterize(page_with_content, dpi=150)

    # 612pt x 792pt at 150 DPI: 612/72*150 = 1275, 792/72*150 = 1650
    assert result.image.shape == (1650, 1275)


def test_rasterize_is_grayscale_uint8(page_with_content) -> None:
    rasterizer = PageRasterizer()
    result = rasterizer.rasterize(page_with_content, dpi=100)

    assert result.image.dtype == np.uint8
    assert result.image.ndim == 2  # single channel - no color/alpha


def test_higher_dpi_produces_proportionally_larger_image(page_with_content) -> None:
    rasterizer = PageRasterizer()
    low = rasterizer.rasterize(page_with_content, dpi=100)
    high = rasterizer.rasterize(page_with_content, dpi=200)

    assert high.image.shape[0] == low.image.shape[0] * 2
    assert high.image.shape[1] == low.image.shape[1] * 2


def test_rasterize_captures_drawn_content(page_with_content) -> None:
    rasterizer = PageRasterizer()
    result = rasterizer.rasterize(page_with_content, dpi=150)

    # A page with a drawn black rectangle on white background must not be
    # uniform - if it were, rendering silently failed or returned blank.
    assert result.image.min() < 50   # the black rectangle
    assert result.image.max() > 200  # the white background


def test_rasterize_records_requested_dpi(page_with_content) -> None:
    rasterizer = PageRasterizer()
    result = rasterizer.rasterize(page_with_content, dpi=300)
    assert result.dpi == 300


def test_rasterize_rejects_nonpositive_dpi(page_with_content) -> None:
    rasterizer = PageRasterizer()
    with pytest.raises(ValueError):
        rasterizer.rasterize(page_with_content, dpi=0)
    with pytest.raises(ValueError):
        rasterizer.rasterize(page_with_content, dpi=-100)
