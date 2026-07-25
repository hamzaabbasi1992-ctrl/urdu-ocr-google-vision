"""Evaluation & Benchmarking: GroundTruthLoader.

Single responsibility: load a reference (known-correct) transcription for
a test page/document. Data only - no comparison, no scoring, no OCR.
"""

from __future__ import annotations

from pathlib import Path


class GroundTruthLoader:
    def load(self, path: Path) -> str:
        return Path(path).read_text(encoding="utf-8-sig")
