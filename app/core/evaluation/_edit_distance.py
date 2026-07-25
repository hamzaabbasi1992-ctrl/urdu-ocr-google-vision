"""Shared Levenshtein edit-distance implementation used by both
CERCalculator (character sequences) and WERCalculator (word sequences).
Not a public module - it has no independent responsibility of its own,
it's the one piece of arithmetic both calculators would otherwise
duplicate identically.
"""

from __future__ import annotations

from collections.abc import Sequence


def levenshtein_distance(a: Sequence, b: Sequence) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current_row = [i] + [0] * len(b)
        for j, item_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            substitute_cost = previous_row[j - 1] + (0 if item_a == item_b else 1)
            current_row[j] = min(insert_cost, delete_cost, substitute_cost)
        previous_row = current_row
    return previous_row[-1]
