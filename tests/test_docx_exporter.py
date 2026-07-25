"""Standalone test for Layer 10's DocxExporter - verifies it writes a
valid .docx with the right text content and RTL formatting applied."""

from __future__ import annotations

from pathlib import Path

from docx import Document

from app.core.export.docx_exporter import DocxExporter


def test_export_creates_readable_docx_with_correct_paragraph_text(tmp_path: Path) -> None:
    path = tmp_path / "output.docx"
    DocxExporter().export("بسم الله\nالحمد للہ", path)

    assert path.exists()
    document = Document(str(path))
    paragraph_texts = [p.text for p in document.paragraphs if p.text.strip()]
    assert paragraph_texts == ["بسم الله", "الحمد للہ"]


def test_export_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "output.docx"
    DocxExporter().export("line one\n\nline two", path)

    document = Document(str(path))
    paragraph_texts = [p.text for p in document.paragraphs if p.text.strip()]
    assert paragraph_texts == ["line one", "line two"]


def test_export_sets_rtl_bidi_on_paragraphs(tmp_path: Path) -> None:
    from docx.oxml.ns import qn

    path = tmp_path / "output.docx"
    DocxExporter().export("متن", path)

    document = Document(str(path))
    paragraph = next(p for p in document.paragraphs if p.text.strip())
    pPr = paragraph._p.pPr
    assert pPr is not None
    bidi = pPr.find(qn("w:bidi"))
    assert bidi is not None, "expected a w:bidi element marking the paragraph as right-to-left"


def test_export_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "output.docx"
    DocxExporter().export("text", path)
    assert path.exists()


def test_export_empty_text_produces_valid_empty_document(tmp_path: Path) -> None:
    path = tmp_path / "output.docx"
    DocxExporter().export("", path)

    assert path.exists()
    document = Document(str(path))
    assert all(not p.text.strip() for p in document.paragraphs)
