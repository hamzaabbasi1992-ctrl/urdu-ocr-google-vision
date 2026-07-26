"""Standalone test for Layer 10's heading_classifier - verifies heading
detection from purely geometric signals (relative line height, isolation
via extra vertical whitespace, and a required shortness check), since
Google Vision exposes neither font weight nor any other direct "is this
bold" signal.

Real-book test (PROJECT_SPEC.md, 2026-07-26) found height/isolation alone
produced a 9.3% false-positive rate on footnotes and mid-paragraph
fragments - both can be tall or isolated but are never short like a real
title, hence the shortness requirement these tests exercise below."""

from __future__ import annotations

from app.core.export.heading_classifier import classify_headings
from app.core.recognition.recognized_word import RecognizedWord


def _line(height: float, width: float = 20.0, y0: float = 0.0) -> list[RecognizedWord]:
    """A single-word line spanning x0=0..width at the given height/y0."""
    return [RecognizedWord(text="w", confidence=1.0, x0=0, y0=y0, x1=width, y1=y0 + height, line_index=0)]


def _empty_line() -> list[RecognizedWord]:
    return []


# ---- height signal (combined with the shortness gate) -------------------


def test_much_taller_short_line_among_uniform_body_lines_is_a_heading() -> None:
    lines = [_line(30, width=20), _line(10, width=100), _line(10, width=100), _line(10, width=100)]
    assert classify_headings(lines) == [True, False, False, False]


def test_uniform_height_lines_are_never_headings() -> None:
    lines = [_line(10, width=100), _line(10, width=100), _line(10, width=100)]
    assert classify_headings(lines) == [False, False, False]


def test_slightly_taller_line_within_normal_variance_is_not_a_heading() -> None:
    # Just under the 1.5x ratio - normal body-text height variance, not a heading.
    lines = [_line(14, width=20), _line(10, width=100), _line(10, width=100), _line(10, width=100)]
    assert classify_headings(lines) == [False, False, False, False]


def test_line_with_no_words_is_never_a_heading() -> None:
    lines = [_line(10, width=100), _empty_line(), _line(10, width=100)]
    assert classify_headings(lines) == [False, False, False]


def test_all_lines_zero_height_returns_no_headings() -> None:
    lines = [_line(0, width=100), _line(0, width=100)]
    assert classify_headings(lines) == [False, False]


def test_tall_but_full_width_line_is_not_a_heading() -> None:
    # A tall line that still spans the full body-line width (e.g. large
    # body text, not a title) must not be classified as a heading - the
    # exact false-positive shape found in the real-book test.
    lines = [_line(30, width=100), _line(10, width=100), _line(10, width=100), _line(10, width=100)]
    assert classify_headings(lines) == [False, False, False, False]


# ---- isolation signal (combined with the shortness gate) ----------------


def test_isolated_short_line_with_large_gap_above_is_a_heading() -> None:
    lines = [
        _line(10, width=100, y0=0),
        _line(10, width=100, y0=15),  # gap above = 5
        _line(10, width=20, y0=50),  # gap above = 25, short - isolated heading
        _line(10, width=100, y0=65),  # gap above = 5
    ]
    assert classify_headings(lines) == [False, False, True, False]


def test_isolated_but_full_width_line_is_not_a_heading() -> None:
    # Same gap pattern as above, but the isolated line is full-width (e.g.
    # a footnote paragraph set apart by whitespace, not a title) - the
    # exact false-positive shape found in the real-book test.
    lines = [
        _line(10, width=100, y0=0),
        _line(10, width=100, y0=15),
        _line(10, width=100, y0=50),  # isolated but full-width
        _line(10, width=100, y0=65),
    ]
    assert classify_headings(lines) == [False, False, False, False]


def test_uniform_line_spacing_never_triggers_isolation() -> None:
    lines = [_line(10, width=100, y0=0), _line(10, width=100, y0=15), _line(10, width=100, y0=30), _line(10, width=100, y0=45)]
    assert classify_headings(lines) == [False, False, False, False]


def test_first_line_cannot_be_classified_a_heading_via_isolation() -> None:
    # No previous line to measure a gap against - only the height signal
    # can mark the very first line, never isolation.
    lines = [_line(10, width=20, y0=0), _line(10, width=100, y0=15), _line(10, width=100, y0=30)]
    assert classify_headings(lines)[0] is False


# ---- combined / empty ----------------------------------------------------


def test_empty_lines_list_returns_empty_result() -> None:
    assert classify_headings([]) == []
