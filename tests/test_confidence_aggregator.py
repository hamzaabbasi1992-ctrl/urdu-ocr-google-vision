"""Standalone test for the Evaluation group's ConfidenceAggregator."""

from __future__ import annotations

import pytest

from app.core.evaluation.confidence_aggregator import ConfidenceAggregator


def test_average_of_multiple_values() -> None:
    assert ConfidenceAggregator().average([0.5, 0.7, 0.9]) == pytest.approx(0.7)


def test_average_of_single_value() -> None:
    assert ConfidenceAggregator().average([0.42]) == pytest.approx(0.42)


def test_average_of_empty_iterable_is_zero() -> None:
    assert ConfidenceAggregator().average([]) == 0.0


def test_accepts_a_generator_not_just_a_list() -> None:
    assert ConfidenceAggregator().average(x / 10 for x in range(1, 5)) == pytest.approx(0.25)
