"""Standalone test for Layer 10's SearchablePDFExporter - verifies the
produced PDF is visually the original page image with a real, searchable
invisible text layer over it, using the real font-lookup path (no mocking)
since this project runs on Windows machines that have Urdu fonts installed."""

from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
import pytest

from app.core.export.searchable_pdf_exporter import SearchablePDFExporter
from app.core.recognition.recognized_word import RecognizedWord


def _blank_page_image(width: int = 200, height: int = 100) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


def test_add_page_embeds_the_original_image(tmp_path: Path) -> None:
    path = tmp_path / "output.pdf"
    exporter = SearchablePDFExporter()
    exporter.add_page(_blank_page_image(), words=[], dpi=100)
    exporter.save(path)

    doc = fitz.open(str(path))
    assert doc.page_count == 1
    assert len(doc[0].get_images()) == 1
    doc.close()


def test_page_size_matches_image_size_at_given_dpi(tmp_path: Path) -> None:
    path = tmp_path / "output.pdf"
    exporter = SearchablePDFExporter()
    exporter.add_page(_blank_page_image(width=200, height=100), words=[], dpi=100)
    exporter.save(path)

    doc = fitz.open(str(path))
    rect = doc[0].rect
    assert rect.width == pytest.approx(200 * 72 / 100)
    assert rect.height == pytest.approx(100 * 72 / 100)
    doc.close()


def test_recognized_words_are_searchable_in_the_output(tmp_path: Path) -> None:
    path = tmp_path / "output.pdf"
    words = [
        RecognizedWord(text="علم", confidence=0.9, x0=10, y0=10, x1=60, y1=30),
        RecognizedWord(text="حاصل", confidence=0.9, x0=70, y0=10, x1=140, y1=30),
    ]
    exporter = SearchablePDFExporter()
    exporter.add_page(_blank_page_image(), words=words, dpi=100)
    exporter.save(path)

    doc = fitz.open(str(path))
    page_text = doc[0].get_text()
    assert "علم" in page_text
    assert "حاصل" in page_text
    doc.close()


def test_invisible_text_layer_does_not_alter_visible_rendering(tmp_path: Path) -> None:
    """The text layer must be invisible (render_mode=3) - the rendered page
    pixmap should match a page with no words added at all, since only the
    background image is meant to be visible."""
    path_with_words = tmp_path / "with_words.pdf"
    path_without_words = tmp_path / "without_words.pdf"
    image = _blank_page_image()

    exporter = SearchablePDFExporter()
    exporter.add_page(image, words=[RecognizedWord(text="علم", confidence=0.9, x0=10, y0=10, x1=60, y1=30)], dpi=100)
    exporter.save(path_with_words)

    exporter2 = SearchablePDFExporter()
    exporter2.add_page(image, words=[], dpi=100)
    exporter2.save(path_without_words)

    doc_a = fitz.open(str(path_with_words))
    doc_b = fitz.open(str(path_without_words))
    pix_a = doc_a[0].get_pixmap()
    pix_b = doc_b[0].get_pixmap()
    assert pix_a.samples == pix_b.samples
    doc_a.close()
    doc_b.close()


def test_blank_words_list_produces_no_text_layer_but_still_saves(tmp_path: Path) -> None:
    path = tmp_path / "output.pdf"
    exporter = SearchablePDFExporter()
    exporter.add_page(_blank_page_image(), words=[], dpi=100)
    exporter.save(path)

    doc = fitz.open(str(path))
    assert doc[0].get_text().strip() == ""
    doc.close()


def test_multiple_pages_accumulate_in_order(tmp_path: Path) -> None:
    path = tmp_path / "output.pdf"
    exporter = SearchablePDFExporter()
    exporter.add_page(
        _blank_page_image(), words=[RecognizedWord(text="اول", confidence=0.9, x0=10, y0=10, x1=60, y1=30)], dpi=100
    )
    exporter.add_page(
        _blank_page_image(), words=[RecognizedWord(text="دوم", confidence=0.9, x0=10, y0=10, x1=60, y1=30)], dpi=100
    )
    exporter.save(path)

    doc = fitz.open(str(path))
    assert doc.page_count == 2
    assert "اول" in doc[0].get_text()
    assert "دوم" in doc[1].get_text()
    doc.close()


def test_reopening_existing_path_appends_pages_for_resume(tmp_path: Path) -> None:
    path = tmp_path / "output.pdf"
    first = SearchablePDFExporter()
    first.add_page(
        _blank_page_image(), words=[RecognizedWord(text="اول", confidence=0.9, x0=10, y0=10, x1=60, y1=30)], dpi=100
    )
    first.save(path)
    first.close()

    resumed = SearchablePDFExporter(existing_path=path)
    resumed.add_page(
        _blank_page_image(), words=[RecognizedWord(text="دوم", confidence=0.9, x0=10, y0=10, x1=60, y1=30)], dpi=100
    )
    resumed.save(path)

    doc = fitz.open(str(path))
    assert doc.page_count == 2
    assert "اول" in doc[0].get_text()
    assert "دوم" in doc[1].get_text()
    doc.close()


def test_empty_word_text_is_skipped_without_error(tmp_path: Path) -> None:
    path = tmp_path / "output.pdf"
    exporter = SearchablePDFExporter()
    exporter.add_page(
        _blank_page_image(),
        words=[RecognizedWord(text="   ", confidence=0.9, x0=10, y0=10, x1=60, y1=30)],
        dpi=100,
    )
    exporter.save(path)
    assert path.exists()
