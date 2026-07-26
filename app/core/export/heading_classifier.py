"""Layer 10 - Export: heading detection for DocxExporter.

Single responsibility: given recognized words already grouped into
reading-order lines, decide which lines are headings, purely from relative
line height (text size). Never inspects or judges the recognized text
itself - this is a formatting decision about already-final, already-correct
text, not a guess about what the text says (PROJECT_SPEC.md Section 2 rule 2
governs the words, not how a line this tall gets styled in Word).
"""

from __future__ import annotations

from app.core.recognition.recognized_word import RecognizedWord

# A line classifies as a heading when its median word height is at least
# this many times the page/region's own median line height. Conservative on
# purpose - body text naturally varies somewhat in height (ascenders,
# descenders, OCR bounding-box noise); only text clearly larger than normal
# body text (chapter/section titles) should cross this bar. Not yet
# validated against a real scanned book's actual heading sizes (see
# PROJECT_SPEC.md Section 9) - a starting value, not a measured one.
HEADING_HEIGHT_RATIO = 1.5


def classify_headings(lines: list[list[RecognizedWord]]) -> list[bool]:
    """Returns one bool per line (same length/order as `lines`)."""
    heights = [_median_height(line) for line in lines]
    positive = [h for h in heights if h > 0]
    if not positive:
        return [False] * len(lines)
    baseline = sorted(positive)[len(positive) // 2]
    if baseline <= 0:
        return [False] * len(lines)
    return [h >= baseline * HEADING_HEIGHT_RATIO for h in heights]


def _median_height(line: list[RecognizedWord]) -> float:
    heights = [w.y1 - w.y0 for w in line]
    if not heights:
        return 0.0
    return sorted(heights)[len(heights) // 2]
