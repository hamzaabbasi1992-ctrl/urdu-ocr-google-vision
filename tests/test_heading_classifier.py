"""Standalone test for Layer 10's heading_classifier - verifies heading
detection is purely a function of relative line height."""

from __future__ import annotations

from app.core.export.heading_classifier import classify_headings
from app.core.recognition.recognized_word import RecognizedWord


def _line(height: float, count: int = 1) -> list[RecognizedWord]:
    return [
        RecognizedWord(text=f"w{i}", confidence=1.0, x0=0, y0=0, x1=1, y1=height, line_index=0)
        for i in range(count)
    ]


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


def test_empty_lines_list_returns_empty_result() -> None:
    assert classify_headings([]) == []


def test_line_with_no_words_is_never_a_heading() -> None:
    lines = [_line(10), [], _line(10)]
    assert classify_headings(lines) == [False, False, False]


def test_all_lines_zero_height_returns_no_headings() -> None:
    lines = [_line(0), _line(0)]
    assert classify_headings(lines) == [False, False]
