"""Standalone test for the shared RecognizedWord/assign_reading_order logic
(Layer 6 - Recognition Engines). Split out of test_paddle_ocr_engine.py when
PaddleOCREngine was removed - reading-order assembly is engine-agnostic and
still used by GoogleVisionEngine."""

from __future__ import annotations

from app.core.recognition.recognized_word import RecognizedWord, assign_reading_order


def test_assign_reading_order_groups_lines_and_sorts_rtl() -> None:
    # Two lines; within each line, words given left-to-right in x but must
    # come back sorted right-to-left (higher x0 first).
    words = [
        RecognizedWord(text="left1", confidence=1.0, x0=10, y0=10, x1=40, y1=30),
        RecognizedWord(text="right1", confidence=1.0, x0=100, y0=10, x1=130, y1=30),
        RecognizedWord(text="left2", confidence=1.0, x0=10, y0=100, x1=40, y1=120),
        RecognizedWord(text="right2", confidence=1.0, x0=100, y0=100, x1=130, y1=120),
    ]
    assign_reading_order(words)

    # First line (smaller y) comes first; within it, "right1" (higher x0) before "left1"
    assert [w.text for w in words] == ["right1", "left1", "right2", "left2"]
    assert words[0].line_index == words[1].line_index
    assert words[2].line_index == words[3].line_index
    assert words[0].line_index != words[2].line_index


def test_assign_reading_order_empty_list_does_not_raise() -> None:
    words: list[RecognizedWord] = []
    assign_reading_order(words)  # must not raise
    assert words == []
