"""Standalone test for the Evaluation group's WERCalculator."""

from __future__ import annotations

import pytest

from app.core.evaluation.wer_calculator import WERCalculator


def test_identical_text_has_zero_wer() -> None:
    assert WERCalculator().calculate("the quick fox", "the quick fox") == 0.0


def test_one_word_substitution() -> None:
    wer = WERCalculator().calculate("the slow fox", "the quick fox")
    assert wer == pytest.approx(1 / 3)


def test_one_word_deletion() -> None:
    wer = WERCalculator().calculate("the fox", "the quick fox")
    assert wer == pytest.approx(1 / 3)


def test_one_word_insertion() -> None:
    wer = WERCalculator().calculate("the very quick fox", "the quick fox")
    assert wer == pytest.approx(1 / 3)


def test_empty_hypothesis_against_nonempty_reference_is_full_deletion() -> None:
    assert WERCalculator().calculate("", "the quick fox") == pytest.approx(1.0)


def test_empty_reference_and_empty_hypothesis_is_zero() -> None:
    assert WERCalculator().calculate("", "") == 0.0


def test_empty_reference_and_nonempty_hypothesis_is_one() -> None:
    assert WERCalculator().calculate("some words", "") == 1.0


def test_urdu_word_level_substitution() -> None:
    reference = "الحمد للہ رب العالمین"
    hypothesis = "الحمد للہ رب العلمین"  # one word altered
    wer = WERCalculator().calculate(hypothesis, reference)
    assert wer == pytest.approx(1 / 4)


def test_whitespace_variations_are_treated_as_word_splits_not_extra_chars() -> None:
    # Multiple spaces between words must not be treated as separate "words"
    wer = WERCalculator().calculate("the   quick  fox", "the quick fox")
    assert wer == 0.0
