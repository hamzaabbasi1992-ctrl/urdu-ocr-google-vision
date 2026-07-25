"""Searchable PDF: the original page image with an invisible Unicode text
layer (PyMuPDF render_mode=3) placed at each word's bounding box, so the
PDF is searchable/copyable while looking exactly like the scan.

Word boxes are stored (see job_queue.py) in the *canonical* per-page image's
coordinate space - orientation-corrected, pre-crop/scale - and
PageResult.raw_image_path points at that same canonical image, so pixel
coordinates convert to PDF points with a single dpi-based scale factor.

Placing invisible-but-extractable Arabic-script text requires a font that
actually has Arabic glyphs (MuPDF needs the font's cmap to encode the
text, even though it's never drawn) - Tahoma ships on every Windows
install and covers the Urdu/Arabic block, so it's used as the layer font.
If no such font can be found, the page still exports as an image-only PDF
(no crash) with a logged warning rather than a silent, wrong text layer.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from app.core.models import DocumentResult

_LOGGER = logging.getLogger("urdu_ocr.exporters.searchable_pdf")

_CANDIDATE_FONTS = [
    Path(r"C:\Windows\Fonts\tahoma.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
]


def _find_arabic_capable_font() -> Path | None:
    for candidate in _CANDIDATE_FONTS:
        if candidate.exists():
            return candidate
    return None


def export_searchable_pdf(result: DocumentResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_path = _find_arabic_capable_font()
    if font_path is None:
        _LOGGER.warning("No Arabic-capable font found on this system; exporting image-only PDF (no text layer).")

    doc = fitz.open()
    try:
        for page in result.pages:
            if page.raw_image_path is None or not page.raw_image_path.exists():
                _LOGGER.warning("Page %d has no stored image; skipping in searchable PDF.", page.page_number)
                continue

            points_per_px = 72.0 / page.chosen_dpi
            page_width_pt = page.width * points_per_px
            page_height_pt = page.height * points_per_px

            pdf_page = doc.new_page(width=page_width_pt, height=page_height_pt)
            pdf_page.insert_image(pdf_page.rect, filename=str(page.raw_image_path))

            if font_path is None:
                continue

            for word in page.words:
                x0 = word.bbox.x0 * points_per_px
                y0 = word.bbox.y0 * points_per_px
                y1 = word.bbox.y1 * points_per_px
                font_size = max(1.0, (y1 - y0) * 0.85)
                try:
                    pdf_page.insert_text(
                        (x0, y1),
                        word.text,
                        fontsize=font_size,
                        fontfile=str(font_path),
                        fontname="urdu-ocr-layer",
                        render_mode=3,  # invisible
                    )
                except Exception as exc:  # noqa: BLE001 - one bad glyph must not abort the whole export
                    _LOGGER.debug("Skipped word in searchable-PDF layer (%s): %r", exc, word.text)

        doc.save(str(output_path))
    finally:
        doc.close()
