"""Layer 10 - Export: SearchablePDFExporter.

Single responsibility: produce a searchable PDF that looks exactly like the
original scanned page (the same raster image, full page, nothing redrawn)
with an invisible OCR text layer positioned over each recognized word - so
the PDF is visually identical to the source scan but the text is
selectable/searchable/copyable, the standard "OCR to searchable PDF" idea
(same approach tools like OCRmyPDF use: image visible, text invisible).

Never renders the recognized text visibly (render_mode=3 - invisible fill
and stroke) - only the original page image is visible. This keeps the
module consistent with PROJECT_SPEC.md Section 2 rule 5 (output must
contain only the OCR result, nothing synthesized) - the "output" here is a
copy/search layer, not a replacement rendering of the page.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np

from app.core.recognition.recognized_word import RecognizedWord

# Common Windows fonts checked in this order for Arabic/Urdu glyph coverage.
# Only used to build the invisible text layer's glyph/ToUnicode mapping, so
# visual design doesn't matter - only Unicode-block coverage does. A real
# Nastaleeq font is tried first since it's most likely to also cover the
# Quranic marks/diacritics/Urdu-Indic numerals this project must never drop
# (PROJECT_SPEC.md Section 2 rule 3), which the generic UI fonts below it
# may not fully include.
_CANDIDATE_FONT_PATHS = [
    Path(r"C:\Windows\Fonts\Jameel Noori Nastaleeq .ttf"),
    Path(r"C:\Windows\Fonts\tahoma.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\times.ttf"),
]

_PROBE_CHAR = ord("ا")  # Arabic letter alef - present in virtually all Urdu text


def _find_urdu_font_path() -> Path:
    for path in _CANDIDATE_FONT_PATHS:
        if not path.exists():
            continue
        if fitz.Font(fontfile=str(path)).has_glyph(_PROBE_CHAR):
            return path
    raise RuntimeError(
        "No installed font with Arabic/Urdu glyph coverage found among: "
        + ", ".join(str(p) for p in _CANDIDATE_FONT_PATHS)
        + " - SearchablePDFExporter needs one to build a searchable text layer."
    )


class SearchablePDFExporter:
    """Builds a multi-page searchable PDF one page at a time.

    Call add_page() once per recognized page (its raster image plus the
    words recognized on it, at the DPI that image was rasterized at), then
    save(). Passing an existing_path whose file already exists reopens it
    and appends further pages to it - lets a checkpointed/resumed run
    continue the same PDF instead of starting over, mirroring the
    checkpoint/resume behavior TextExporter/DocxExporter already get from
    app/simple_gui.py.
    """

    def __init__(self, existing_path: Path | None = None) -> None:
        if existing_path is not None and existing_path.exists():
            # Opened from an in-memory byte buffer, not the path directly -
            # opening straight from the path would keep a lock/mmap on that
            # file for the life of this object, which would then fail when
            # save() tries to atomically replace that same path.
            self._doc = fitz.open(stream=existing_path.read_bytes(), filetype="pdf")
        else:
            self._doc = fitz.open()
        self._font_path = str(_find_urdu_font_path())

    @property
    def page_count(self) -> int:
        return self._doc.page_count

    def add_page(self, image: np.ndarray, words: list[RecognizedWord], dpi: int) -> None:
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Could not encode page image for the searchable PDF")

        scale = 72.0 / dpi
        height, width = image.shape[:2]
        page_rect = fitz.Rect(0, 0, width * scale, height * scale)
        page = self._doc.new_page(width=page_rect.width, height=page_rect.height)
        page.insert_image(page_rect, stream=encoded.tobytes())

        text_words = [w for w in words if w.text.strip()]
        if not text_words:
            return

        page.insert_font(fontname="urdu-searchlayer", fontfile=self._font_path)
        for word in text_words:
            x0, y0, x1, y1 = (word.x0 * scale, word.y0 * scale, word.x1 * scale, word.y1 * scale)
            fontsize = max((y1 - y0) * 0.9, 1.0)
            origin = fitz.Point(x0, y1 - (y1 - y0) * 0.15)  # roughly the visual baseline
            # insert_text lays characters out left-to-right by glyph advance
            # regardless of script - it does not apply bidi/RTL reordering.
            # For a right-to-left word that places the glyphs in mirrored
            # position order, which MuPDF's own text-extraction bidi logic
            # then flips again on read-back, turning "علم" into "ملع".
            # Reversing the character order before insertion cancels that
            # out, so get_text() reconstructs the original word correctly -
            # verified directly against real Urdu text, not assumed.
            page.insert_text(
                origin,
                word.text[::-1],
                fontsize=fontsize,
                fontname="urdu-searchlayer",
                render_mode=3,  # invisible - only the page image is seen
            )

    def save(self, path: Path) -> None:
        """Writes the accumulated document to `path`, atomically (temp file
        then replace) so a crash mid-write never leaves a corrupt PDF at the
        real output path - the same safety property the checkpoint interval
        already relies on for TXT/DOCX."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        self._doc.save(str(tmp_path))
        tmp_path.replace(path)

    def close(self) -> None:
        self._doc.close()
