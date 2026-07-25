"""Evaluation & Benchmarking: ConfidenceAggregator.

Single responsibility: given a set of OCR word confidences, compute an
average. Pure function.
"""

from __future__ import annotations

from collections.abc import Iterable


class ConfidenceAggregator:
    def average(self, confidences: Iterable[float]) -> float:
        confidences = list(confidences)
        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)
