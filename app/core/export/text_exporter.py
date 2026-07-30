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
from app.core.postprocessing.low_confidence_flagger import flag_low_confidence
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


# Wrap a low-confidence word's text between these two (see
# assemble_text_with_headings) - mathematical white square brackets, chosen
# because they don't occur in real Urdu/Arabic OCR text, so they can't be
# confused with genuine punctuation. Unlike HEADING_MARKER these are left
# VISIBLE in output.txt on purpose: Section 2 rule 2 requires low-confidence
# recognition to be flagged, and plain .txt has no other way to carry that
# signal. DocxExporter converts them into a highlighted run instead of
# visible brackets, since .docx has real formatting to carry the same flag.
LOW_CONFIDENCE_START = "⟦"
LOW_CONFIDENCE_END = "⟧"


def _group_lines(words: list[RecognizedWord]) -> list[list[RecognizedWord]]:
    lines: dict[int, list[RecognizedWord]] = {}
    for word in words:
        lines.setdefault(word.line_index, []).append(word)
    return [lines[i] for i in sorted(lines)]


def _join_line(line_words: list[RecognizedWord], flags: list[bool] | None = None) -> str:
    parts: list[str] = []
    flags = flags or [False] * len(line_words)
    for word, flagged in zip(line_words, flags):
        text = f"{LOW_CONFIDENCE_START}{word.text}{LOW_CONFIDENCE_END}" if flagged else word.text
        if parts and word.text in _NO_LEADING_SPACE:
            parts[-1] += text
        else:
            parts.append(text)
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
    line height) with HEADING_MARKER, and wraps each word flagged as
    low-confidence (by app.core.postprocessing.low_confidence_flagger,
    purely from the engine's own per-word confidence score) between
    LOW_CONFIDENCE_START/END."""
    if not words:
        return ""
    lines = _group_lines(words)
    is_heading = classify_headings(lines)
    low_confidence = flag_low_confidence(words)
    line_flags = _group_lines_of(low_confidence, words)
    text_lines = [
        (HEADING_MARKER if heading else "") + _join_line(line, flags)
        for line, heading, flags in zip(lines, is_heading, line_flags)
    ]
    return "\n".join(text_lines)


def _group_lines_of(values: list[bool], words: list[RecognizedWord]) -> list[list[bool]]:
    """Regroups a flat per-word value list (aligned with `words`) using the
    same line_index grouping _group_lines uses, so a value list computed
    once over the flat word order (as flag_low_confidence's contract
    requires) can be zipped against the already-grouped `lines`."""
    grouped: dict[int, list[bool]] = {}
    for word, value in zip(words, values):
        grouped.setdefault(word.line_index, []).append(value)
    return [grouped[i] for i in sorted(grouped)]


class TextExporter:
    def export(self, text: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        stripped = "\n".join(line.removeprefix(HEADING_MARKER) for line in text.split("\n"))
        path.write_text(stripped, encoding="utf-8-sig")
