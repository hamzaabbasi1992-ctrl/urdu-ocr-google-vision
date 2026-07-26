"""Layer 10 - Export: heading detection for DocxExporter.

Single responsibility: given recognized words already grouped into
reading-order lines, decide which lines are headings. Never inspects or
judges the recognized text itself - this is a formatting decision about
already-final, already-correct text, not a guess about what the text says
(PROJECT_SPEC.md Section 2 rule 2 governs the words, not how a line like
this gets styled in Word).

Two independent signals, either one sufficient to classify a line as a
heading:
1. Relative height - the line's text is noticeably taller than typical body
   text.
2. Isolation - the line has noticeably more vertical whitespace above it
   than typical line-to-line spacing (a blank-line-separated heading, even
   one that isn't itself taller than body text).

Google Vision's DOCUMENT_TEXT_DETECTION response does not expose font
weight/style (checked directly against the google-cloud-vision client's
Word/Paragraph/Block/Symbol/TextProperty protobuf fields - only
bounding_box, confidence, and language/line-break info exist) - "bold"
cannot be read from that data at all, only inferred indirectly via these
geometric signals derived from bounding boxes.
"""

from __future__ import annotations

from app.core.recognition.recognized_word import RecognizedWord

# A line classifies as a heading when its median word height is at least
# this many times the page/region's own median line height. Conservative on
# purpose - body text naturally varies somewhat in height (ascenders,
# descenders, OCR bounding-box noise); only text clearly larger than normal
# body text (chapter/section titles) should cross this bar.
HEADING_HEIGHT_RATIO = 1.5

# A line classifies as a heading when the vertical gap above it is at least
# this many times the page/region's own typical line-to-line gap - catches
# headings that are set apart by whitespace even when not themselves taller
# than body text (e.g. bold-only headings, which height alone can't see).
HEADING_GAP_RATIO = 2.0

# Neither ratio above is measured against a real book yet - both are
# starting heuristics pending real-world validation (see PROJECT_SPEC.md).


def classify_headings(lines: list[list[RecognizedWord]]) -> list[bool]:
    """Returns one bool per line (same length/order as `lines`)."""
    if not lines:
        return []

    heights = [_median_height(line) for line in lines]
    height_baseline = _median_positive(heights)

    gaps = _gaps_above(lines)
    gap_baseline = _median_positive(gaps)

    result = []
    for height, gap in zip(heights, gaps):
        is_tall = height_baseline > 0 and height >= height_baseline * HEADING_HEIGHT_RATIO
        is_isolated = gap_baseline > 0 and gap >= gap_baseline * HEADING_GAP_RATIO
        result.append(is_tall or is_isolated)
    return result


def _median_height(line: list[RecognizedWord]) -> float:
    heights = [w.y1 - w.y0 for w in line]
    if not heights:
        return 0.0
    return sorted(heights)[len(heights) // 2]


def _line_top(line: list[RecognizedWord]) -> float:
    ys = [w.y0 for w in line]
    return min(ys) if ys else 0.0


def _line_bottom(line: list[RecognizedWord]) -> float:
    ys = [w.y1 for w in line]
    return max(ys) if ys else 0.0


def _gaps_above(lines: list[list[RecognizedWord]]) -> list[float]:
    """Vertical gap between each line and the one before it. The first line
    has no previous line to measure against, so its gap is 0 (not
    isolated) - a conservative default, not a guess."""
    gaps = [0.0]
    for i in range(1, len(lines)):
        gaps.append(max(0.0, _line_top(lines[i]) - _line_bottom(lines[i - 1])))
    return gaps


def _median_positive(values: list[float]) -> float:
    positive = [v for v in values if v > 0]
    if not positive:
        return 0.0
    return sorted(positive)[len(positive) // 2]
