"""Step 5: purely mechanical text assembly. No language model, no spelling
or grammar correction, no sentence generation - every character an engine
recognized (punctuation, Urdu-Indic numerals, Arabic loanwords, diacritics)
passes through untouched. The only operations here are whitespace cleanup
and re-joining words/lines that OCR fragmented; nothing is invented and
nothing recognized is discarded.

Deliberately never calls unicodedata.normalize or any casing/composition
transform - either of those can silently strip or reorder combining
diacritic marks (zabar/zer/pesh, etc.), which the project rules forbid.
"""

from __future__ import annotations

import re

from app.core.models import OCRWord
from app.core.ocr.engine_base import assign_line_indices

_WHITESPACE_RE = re.compile(r"[ \t]+")


def postprocess_page(words: list[OCRWord]) -> str:
    """Builds a page's text from its recognized words. Reclusters into
    lines by vertical position (rather than trusting each word's existing
    line_index) because merged multi-engine results (see arbiter.py) can
    carry line numbers from two independent, non-comparable numbering
    schemes - this is the "merge broken lines" step in practice."""
    if not words:
        return ""

    assign_line_indices(words)

    lines: dict[int, list[OCRWord]] = {}
    for word in words:
        lines.setdefault(word.line_index, []).append(word)

    text_lines = []
    for line_index in sorted(lines):
        line_words = lines[line_index]  # already right-to-left ordered by assign_line_indices
        line_text = " ".join(w.text for w in line_words)
        line_text = _WHITESPACE_RE.sub(" ", line_text).strip()
        if line_text:
            text_lines.append(line_text)

    return "\n".join(text_lines)


def low_confidence_report(words: list[OCRWord]) -> list[OCRWord]:
    """Returns the flagged low-confidence words for a page, for the GUI to
    highlight - never altered, just surfaced."""
    return [w for w in words if w.low_confidence]
