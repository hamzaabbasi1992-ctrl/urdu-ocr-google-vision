"""Layer 9 - Post-Processing: LowConfidenceFlagger.

Single responsibility: given recognized words, decide which ones are below
a confidence threshold. Flags, never edits - this exists because
PROJECT_SPEC.md Section 2 rule 2 requires low-confidence recognition to be
flagged rather than silently guessed at or silently trusted; a word's
recognized text is never touched here, only its flagged/not-flagged status
is decided.
"""

from __future__ import annotations

from app.core.recognition.recognized_word import RecognizedWord

# A word is flagged as low-confidence when Google Vision's own per-word
# confidence score is below this. 0.5 is a starting value, not a measured
# one - same status as HEADING_HEIGHT_RATIO/HEADING_GAP_RATIO/
# HEADING_MAX_WIDTH_RATIO in heading_classifier.py: picked by reasoning
# (roughly "the engine itself is less than half-sure"), pending real-book
# validation of how many real recognition errors this threshold actually
# catches vs. how many confidently-correct words it wrongly flags.
LOW_CONFIDENCE_THRESHOLD = 0.5


def flag_low_confidence(words: list[RecognizedWord]) -> list[bool]:
    """Returns one bool per word (same length/order as `words`)."""
    return [word.confidence < LOW_CONFIDENCE_THRESHOLD for word in words]
