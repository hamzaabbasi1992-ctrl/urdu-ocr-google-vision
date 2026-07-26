"""Standalone test for Layer 10's heading_classifier - verifies heading
detection from two independent geometric signals (relative line height,
and isolation via extra vertical whitespace), since Google Vision exposes
neither font weight nor any other direct "is this bold" signal."""

from __future__ import annotations

from app.core.export.heading_classifier import classify_headings
from app.core.recognition.recognized_word import RecognizedWord


def _line(height: float, count: int = 1) -> list[RecognizedWord]:
    """A line with words stacked at y0=0 - useful for height-only tests,
    since every line then has zero measurable gap above it (see
    _line_at for tests that need real vertical positions)."""
    return [
        RecognizedWord(text=f"w{i}", confidence=1.0, x0=0, y0=0, x1=1, y1=height, line_index=0)
        for i in range(count)
    ]


def _line_at(y0: float, y1: float) -> list[RecognizedWord]:
    return [RecognizedWord(text="w", confidence=1.0, x0=0, y0=y0, x1=1, y1=y1, line_index=0)]


# ---- height signal -----------------------------------------------------


def test_much_taller_line_among_uniform_body_lines_is_a_heading() -> None:
    lines = [_line(30), _line(10), _line(10), _line(10)]
    assert classify_headings(lines) == [True, False, False, False]


def test_uniform_height_lines_are_never_headings() -> None:
    lines = [_line(10), _line(10), _line(10)]
    assert classify_headings(lines) == [False, False, False]


def test_slightly_taller_line_within_normal_variance_is_not_a_heading() -> None:
    # Just under the 1.5x ratio - normal body-text height variance, not a heading.
    lines = [_line(14), _line(10), _line(10), _line(10)]
    assert classify_headings(lines) == [False, False, False, False]


def test_line_with_no_words_is_never_a_heading() -> None:
    lines = [_line(10), [], _line(10)]
    assert classify_headings(lines) == [False, False, False]


def test_all_lines_zero_height_returns_no_headings() -> None:
    lines = [_line(0), _line(0)]
    assert classify_headings(lines) == [False, False]


# ---- isolation signal (blank space above the line) ----------------------


def test_isolated_line_with_large_gap_above_is_a_heading_even_at_normal_height() -> None:
    # All four lines are the same height (10) - only the gap above line 2
    # (25, vs. a typical ~5 gap elsewhere) should mark it as a heading.
    lines = [
        _line_at(0, 10),
        _line_at(15, 25),  # gap above = 5
        _line_at(50, 60),  # gap above = 25 - isolated
        _line_at(65, 75),  # gap above = 5
    ]
    assert classify_headings(lines) == [False, False, True, False]


def test_uniform_line_spacing_never_triggers_isolation() -> None:
    lines = [_line_at(0, 10), _line_at(15, 25), _line_at(30, 40), _line_at(45, 55)]
    assert classify_headings(lines) == [False, False, False, False]


def test_first_line_cannot_be_classified_a_heading_via_isolation() -> None:
    # No previous line to measure a gap against - only the height signal
    # can mark the very first line, never isolation.
    lines = [_line_at(0, 10), _line_at(15, 25), _line_at(30, 40)]
    assert classify_headings(lines)[0] is False


# ---- combined / empty ----------------------------------------------------


def test_empty_lines_list_returns_empty_result() -> None:
    assert classify_headings([]) == []
