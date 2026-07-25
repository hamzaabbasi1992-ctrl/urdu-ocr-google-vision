"""Standalone test for Layer 2's MultiDPICandidateGenerator - verifies it
renders every requested DPI and nothing more (no scoring, no picking a
winner), and that it purely delegates rendering to PageRasterizer rather
than reimplementing any image work itself."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from app.core.ingestion.page_rasterizer import PageRasterizer, RasterizedPage
from app.core.ingestion.pdf_loader import PDFLoader
from app.core.render_quality.multi_dpi_candidate_generator import MultiDPICandidateGenerator


class _RecordingRasterizer:
    """Stub standing in for PageRasterizer, to prove this module only
    orchestrates calls to it rather than doing any rendering of its own."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def rasterize(self, page: fitz.Page, dpi: int) -> RasterizedPage:
        self.calls.append(dpi)
        return RasterizedPage(dpi=dpi, image=None)  # content doesn't matter for this test


@pytest.fixture
def sample_page(tmp_path: Path) -> fitz.Page:
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    path = tmp_path / "sample.pdf"
    doc.save(path)
    doc.close()

    loader = PDFLoader(path)
    yield loader.get_page(0)
    loader.close()


def test_generate_calls_rasterizer_once_per_dpi_in_order(sample_page: fitz.Page) -> None:
    stub = _RecordingRasterizer()
    generator = MultiDPICandidateGenerator(rasterizer=stub)

    generator.generate(sample_page, dpis=(300, 600, 900))

    assert stub.calls == [300, 600, 900]


def test_generate_returns_one_candidate_per_dpi(sample_page: fitz.Page) -> None:
    stub = _RecordingRasterizer()
    generator = MultiDPICandidateGenerator(rasterizer=stub)

    candidates = generator.generate(sample_page, dpis=(300, 600))

    assert len(candidates) == 2
    assert [c.dpi for c in candidates] == [300, 600]


def test_empty_dpi_list_rejected(sample_page: fitz.Page) -> None:
    generator = MultiDPICandidateGenerator(rasterizer=_RecordingRasterizer())
    with pytest.raises(ValueError):
        generator.generate(sample_page, dpis=())


def test_end_to_end_with_real_rasterizer_produces_increasing_resolutions(sample_page: fitz.Page) -> None:
    """Integration check with the real PageRasterizer - confirms the two
    modules actually work together, not just that the stub contract holds."""
    generator = MultiDPICandidateGenerator(rasterizer=PageRasterizer())

    candidates = generator.generate(sample_page, dpis=(150, 300))

    assert candidates[0].image.shape[0] * 2 == candidates[1].image.shape[0]
    assert candidates[0].image.shape[1] * 2 == candidates[1].image.shape[1]
