"""Standalone test for the Evaluation group's GroundTruthLoader."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.evaluation.ground_truth_loader import GroundTruthLoader


def test_loads_exact_file_content(tmp_path: Path) -> None:
    path = tmp_path / "ground_truth.txt"
    path.write_text("الحمد لله رب العالمین", encoding="utf-8-sig")

    assert GroundTruthLoader().load(path) == "الحمد لله رب العالمین"


def test_handles_bom_transparently(tmp_path: Path) -> None:
    path = tmp_path / "ground_truth.txt"
    path.write_bytes(b"\xef\xbb\xbf" + "بسم اللہ".encode("utf-8"))

    text = GroundTruthLoader().load(path)
    assert not text.startswith("﻿")  # BOM must not leak into the returned string
    assert text == "بسم اللہ"


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        GroundTruthLoader().load(Path("does_not_exist_at_all.txt"))
