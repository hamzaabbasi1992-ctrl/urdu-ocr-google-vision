"""Standalone test for the Evaluation group's CERCalculator - verified
against the standard Wikipedia Levenshtein-distance example and basic
edge cases."""

from __future__ import annotations

import pytest

from app.core.evaluation.cer_calculator import CERCalculator


def test_identical_strings_have_zero_cer() -> None:
    assert CERCalculator().calculate("hello", "hello") == 0.0


def test_kitten_sitting_known_edit_distance() -> None:
    # Canonical example: edit distance("kitten", "sitting") == 3
    cer = CERCalculator().calculate("kitten", "sitting")
    assert cer == pytest.approx(3 / 7)


def test_completely_different_same_length_is_all_substitutions() -> None:
    cer = CERCalculator().calculate("aaaa", "bbbb")
    assert cer == pytest.approx(1.0)


def test_empty_hypothesis_against_nonempty_reference_is_full_deletion() -> None:
    cer = CERCalculator().calculate("", "hello")
    assert cer == pytest.approx(1.0)


def test_empty_reference_and_empty_hypothesis_is_zero() -> None:
    assert CERCalculator().calculate("", "") == 0.0


def test_empty_reference_and_nonempty_hypothesis_is_one() -> None:
    # Defined as 1.0 (maximal error) rather than dividing by zero.
    assert CERCalculator().calculate("hello", "") == 1.0


def test_urdu_text_single_character_substitution() -> None:
    # "کتاب" (book) vs "کتاب" with one character changed -> distance 1
    reference = "کتاب"
    hypothesis = "کتاپ"
    cer = CERCalculator().calculate(hypothesis, reference)
    assert cer == pytest.approx(1 / 4)


def test_insertion_increases_cer() -> None:
    cer = CERCalculator().calculate("helllo", "hello")
    assert cer == pytest.approx(1 / 5)
