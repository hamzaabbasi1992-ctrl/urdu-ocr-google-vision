"""Evaluation & Benchmarking: WERCalculator.

Single responsibility: given hypothesis + reference text, compute Word
Error Rate (Levenshtein edit distance over whitespace-split words /
reference word count). Pure function.
"""

from __future__ import annotations

from app.core.evaluation._edit_distance import levenshtein_distance


class WERCalculator:
    def calculate(self, hypothesis: str, reference: str) -> float:
        reference_words = reference.split()
        hypothesis_words = hypothesis.split()

        if not reference_words:
            return 0.0 if not hypothesis_words else 1.0

        distance = levenshtein_distance(hypothesis_words, reference_words)
        return distance / len(reference_words)
