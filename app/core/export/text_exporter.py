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

from app.core.recognition.recognized_word import RecognizedWord

# Punctuation marks that attach directly to the preceding word with no
# space, matching normal Urdu/Arabic typesetting - some engines (Google
# Vision included) return these as separate word tokens, and naively
# joining every token with a space produces "لفظ ۔" instead of "لفظ۔",
# which is a pure text-assembly artifact, not a recognition difference.
_NO_LEADING_SPACE = set("۔،؟!:؛.,?!;:")


def assemble_text(words: list[RecognizedWord]) -> str:
    """Joins words within a line (attaching trailing punctuation directly,
    space-separating everything else) and lines with a newline, preserving
    the order already present in `words` (does not re-sort)."""
    if not words:
        return ""

    lines: dict[int, list[RecognizedWord]] = {}
    for word in words:
        lines.setdefault(word.line_index, []).append(word)

    text_lines = []
    for i in sorted(lines):
        parts: list[str] = []
        for word in lines[i]:
            if parts and word.text in _NO_LEADING_SPACE:
                parts[-1] += word.text
            else:
                parts.append(word.text)
        text_lines.append(" ".join(parts))

    return "\n".join(text_lines)


class TextExporter:
    def export(self, text: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8-sig")
