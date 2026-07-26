"""Layer 10 - Export: TextExporter.

Single responsibility: produce output.txt from recognized words. Includes
mechanical line assembly (grouping already-ordered words by line and
joining them) since Layer 9 (Post-Processing) is out of scope for the
minimal pipeline (PROJECT_SPEC.md Section 4/Q9) - words already arrive in
reading order via assign_reading_order (called by the recognition engine),
so nothing here reorders or rewrites anything, it only joins what's already
correctly ordered.
"""

from __future__ import annotations

from pathlib import Path

from app.core.export.heading_classifier import classify_headings
from app.core.recognition.recognized_word import RecognizedWord

# Punctuation marks that attach directly to the preceding word with no
# space, matching normal Urdu/Arabic typesetting - some engines (Google
# Vision included) return these as separate word tokens, and naively
# joining every token with a space produces "لفظ ۔" instead of "لفظ۔",
# which is a pure text-assembly artifact, not a recognition difference.
_NO_LEADING_SPACE = set("۔،؟!:؛.,?!;:")

# Unicode Private Use Area character - never produced by real OCR text, so
# it's safe to use as an inline sentinel that tags a line as a heading
# (see assemble_text_with_headings) while riding transparently through the
# existing plain-text accumulation/checkpoint-resume pipeline in
# app/simple_gui.py unchanged. TextExporter strips it before writing
# output.txt (Section 2 rule 5: output must contain only the OCR result);
# DocxExporter strips it too, converting it into a Heading paragraph style
# instead of visible text.
HEADING_MARKER = ""


def _group_lines(words: list[RecognizedWord]) -> list[list[RecognizedWord]]:
    lines: dict[int, list[RecognizedWord]] = {}
    for word in words:
        lines.setdefault(word.line_index, []).append(word)
    return [lines[i] for i in sorted(lines)]


def _join_line(line_words: list[RecognizedWord]) -> str:
    parts: list[str] = []
    for word in line_words:
        if parts and word.text in _NO_LEADING_SPACE:
            parts[-1] += word.text
        else:
            parts.append(word.text)
    return " ".join(parts)


def assemble_text(words: list[RecognizedWord]) -> str:
    """Joins words within a line (attaching trailing punctuation directly,
    space-separating everything else) and lines with a newline, preserving
    the order already present in `words` (does not re-sort)."""
    if not words:
        return ""
    return "\n".join(_join_line(line) for line in _group_lines(words))


def assemble_text_with_headings(words: list[RecognizedWord]) -> str:
    """Same as assemble_text, but prefixes each line classified as a
    heading (by app.core.export.heading_classifier, purely from relative
    line height) with HEADING_MARKER."""
    if not words:
        return ""
    lines = _group_lines(words)
    is_heading = classify_headings(lines)
    text_lines = [
        (HEADING_MARKER if heading else "") + _join_line(line)
        for line, heading in zip(lines, is_heading)
    ]
    return "\n".join(text_lines)


class TextExporter:
    def export(self, text: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        stripped = "\n".join(line.removeprefix(HEADING_MARKER) for line in text.split("\n"))
        path.write_text(stripped, encoding="utf-8-sig")
