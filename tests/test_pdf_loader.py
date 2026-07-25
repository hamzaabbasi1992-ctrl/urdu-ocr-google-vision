"""Standalone test for Layer 1's PDFLoader - verifies its single
responsibility (open a PDF, expose page count/metadata/raw pages) in
isolation, with no dependency on rendering, preprocessing, or OCR."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from app.core.ingestion.pdf_loader import PDFLoader


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """A minimal, real 3-page PDF - no rendering pipeline involved, just
    PyMuPDF creating blank pages directly, to keep this test isolated to
    PDFLoader's own responsibility."""
    doc = fitz.open()
    for _ in range(3):
        doc.new_page(width=612, height=792)  # standard US Letter, in points
    doc.set_metadata({"title": "Test Document", "author": "Test Author"})
    path = tmp_path / "sample.pdf"
    doc.save(path)
    doc.close()
    return path


def test_page_count(sample_pdf: Path) -> None:
    with PDFLoader(sample_pdf) as loader:
        assert loader.page_count == 3


def test_metadata(sample_pdf: Path) -> None:
    with PDFLoader(sample_pdf) as loader:
        meta = loader.metadata()
        assert meta.page_count == 3
        assert meta.title == "Test Document"
        assert meta.author == "Test Author"


def test_get_page_returns_correct_page(sample_pdf: Path) -> None:
    with PDFLoader(sample_pdf) as loader:
        page = loader.get_page(1)
        assert page.number == 1


def test_get_page_out_of_range_raises(sample_pdf: Path) -> None:
    with PDFLoader(sample_pdf) as loader:
        with pytest.raises(IndexError):
            loader.get_page(3)
        with pytest.raises(IndexError):
            loader.get_page(-1)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PDFLoader(tmp_path / "does_not_exist.pdf")


def test_path_property(sample_pdf: Path) -> None:
    with PDFLoader(sample_pdf) as loader:
        assert loader.path == sample_pdf


def test_context_manager_closes_cleanly(sample_pdf: Path) -> None:
    loader = PDFLoader(sample_pdf)
    with loader:
        assert loader.page_count == 3
    # After exit, the underlying document is closed; accessing page_count
    # again should fail rather than silently returning stale data.
    with pytest.raises(ValueError):
        _ = loader.page_count
