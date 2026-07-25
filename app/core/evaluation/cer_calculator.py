"""Evaluation & Benchmarking: CERCalculator.

Single responsibility: given hypothesis + reference text, compute
Character Error Rate (Levenshtein edit distance / reference length). Pure
function - no whitespace normalization or any other text massaging; it
scores exactly the two strings it is given.
"""

from __future__ import annotations

from app.core.evaluation._edit_distance import levenshtein_distance


class CERCalculator:
    def calculate(self, hypothesis: str, reference: str) -> float:
        if not reference:
            return 0.0 if not hypothesis else 1.0
        distance = levenshtein_distance(hypothesis, reference)
        return distance / len(reference)
