"""Layer 10 - Export: DocxExporter.

Single responsibility: produce output.docx from already-assembled text.
Right-to-left paragraphs and runs so Urdu renders and edits naturally in
Word, not as a left-to-right jumble.

RTL-setting approach (bidi paragraph element, run.font.rtl, bidi language
tag) ported from the old app/core/exporters/docx_exporter.py rather than
re-derived - per PROJECT_SPEC.md Section 4/7, that logic was already
correct and tested; only the input shape changed (a plain assembled string
here, not a DocumentResult with Page/OCRWord objects).
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml.shared import OxmlElement
from docx.shared import Pt


def _set_paragraph_rtl(paragraph) -> None:
    pPr = paragraph._p.get_or_add_pPr()
    bidi = OxmlElement("w:bidi")
    pPr.append(bidi)


class DocxExporter:
    def export(self, text: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        document = Document()

        for line in text.split("\n"):
            if not line.strip():
                continue
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            _set_paragraph_rtl(paragraph)
            run = paragraph.add_run(line)
            run.font.rtl = True
            run.font.size = Pt(14)
            rpr = run._element.get_or_add_rPr()
            lang = OxmlElement("w:lang")
            lang.set(qn("w:bidi"), "ur-PK")
            rpr.append(lang)

        document.save(str(path))
