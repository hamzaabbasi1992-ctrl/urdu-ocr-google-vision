"""Standalone test for Layer 9's LowConfidenceFlagger - a pure function over
per-word confidence scores, independent of position/text content."""

from __future__ import annotations

from app.core.postprocessing.low_confidence_flagger import (
    LOW_CONFIDENCE_THRESHOLD,
    flag_low_confidence,
)
from app.core.recognition.recognized_word import RecognizedWord


def _word(confidence: float) -> RecognizedWord:
    return RecognizedWord(text="w", confidence=confidence, x0=0, y0=0, x1=10, y1=10)


def test_word_below_threshold_is_flagged() -> None:
    assert flag_low_confidence([_word(LOW_CONFIDENCE_THRESHOLD - 0.01)]) == [True]


def test_word_at_or_above_threshold_is_not_flagged() -> None:
    assert flag_low_confidence([_word(LOW_CONFIDENCE_THRESHOLD)]) == [False]
    assert flag_low_confidence([_word(LOW_CONFIDENCE_THRESHOLD + 0.1)]) == [False]


def test_high_confidence_word_is_not_flagged() -> None:
    assert flag_low_confidence([_word(0.99)]) == [False]


def test_mixed_confidence_words_flag_only_the_low_ones() -> None:
    words = [_word(0.9), _word(0.1), _word(0.6), _word(0.3)]
    assert flag_low_confidence(words) == [False, True, False, True]


def test_empty_list_returns_empty_list() -> None:
    assert flag_low_confidence([]) == []
