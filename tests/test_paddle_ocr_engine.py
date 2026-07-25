"""Standalone test for Layer 6's PaddleOCREngine.

Split into two groups:
1. Pure-logic tests against fake PaddleOCR result objects (fast, no model
   loading) - covers both the modern (3.x dict-like) and legacy (2.x list)
   result shapes, and the reading-order assembly.
2. One slower integration test that actually constructs the engine and
   runs it on a real rendered-text image, using the already-cached
   PaddleOCR models from earlier sessions (no network needed). This is the
   only proof the constructor/version-fallback logic actually works against
   the real, installed paddleocr/paddlepaddle versions.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.recognition.paddle_ocr_engine import (
    RecognizedWord,
    assign_reading_order,
    parse_paddleocr_result,
)

# ---- pure-logic tests -------------------------------------------------


class _FakeModernPageResult(dict):
    """Modern (3.x) PaddleOCR results are dict-like (rec_texts/rec_scores/
    rec_polys) - a plain dict satisfies that contract for parsing purposes."""


def test_parse_modern_dict_shape() -> None:
    page = _FakeModernPageResult(
        rec_texts=["\u0627\u0644\u0641", "\u0628"],  # "الف", "ب" - arbitrary Urdu/Arabic letters
        rec_scores=[0.9, 0.4],
        rec_polys=[
            [(10, 10), (50, 10), (50, 30), (10, 30)],
            [(60, 10), (90, 10), (90, 30), (60, 30)],
        ],
    )
    words = parse_paddleocr_result([page])

    assert len(words) == 2
    assert {w.text for w in words} == {"\u0627\u0644\u0641", "\u0628"}
    scores = {w.text: w.confidence for w in words}
    assert scores["\u0627\u0644\u0641"] == pytest.approx(0.9)


def test_parse_modern_shape_skips_blank_text() -> None:
    page = _FakeModernPageResult(
        rec_texts=["   ", "\u0628"],
        rec_scores=[0.9, 0.5],
        rec_polys=[[(0, 0), (10, 0), (10, 10), (0, 10)], [(20, 0), (30, 0), (30, 10), (20, 10)]],
    )
    words = parse_paddleocr_result([page])
    assert len(words) == 1
    assert words[0].text == "\u0628"


def test_parse_legacy_list_shape() -> None:
    page = [
        [[(10, 10), (50, 10), (50, 30), (10, 30)], ("\u0627\u0644\u0641", 0.85)],
        [[(60, 10), (90, 10), (90, 30), (60, 30)], ("\u0628", 0.3)],
    ]
    words = parse_paddleocr_result([page])
    assert len(words) == 2
    assert words[0].confidence == pytest.approx(0.85)


def test_parse_handles_garbage_without_raising() -> None:
    # Not a real result shape at all - must degrade to an empty list, not crash.
    words = parse_paddleocr_result([object()])
    assert words == []


def test_parse_empty_page_list() -> None:
    assert parse_paddleocr_result([]) == []


def test_assign_reading_order_groups_lines_and_sorts_rtl() -> None:
    # Two lines; within each line, words given left-to-right in x but must
    # come back sorted right-to-left (higher x0 first).
    words = [
        RecognizedWord(text="left1", confidence=1.0, x0=10, y0=10, x1=40, y1=30),
        RecognizedWord(text="right1", confidence=1.0, x0=100, y0=10, x1=130, y1=30),
        RecognizedWord(text="left2", confidence=1.0, x0=10, y0=100, x1=40, y1=120),
        RecognizedWord(text="right2", confidence=1.0, x0=100, y0=100, x1=130, y1=120),
    ]
    assign_reading_order(words)

    # First line (smaller y) comes first; within it, "right1" (higher x0) before "left1"
    assert [w.text for w in words] == ["right1", "left1", "right2", "left2"]
    assert words[0].line_index == words[1].line_index
    assert words[2].line_index == words[3].line_index
    assert words[0].line_index != words[2].line_index


def test_assign_reading_order_empty_list_does_not_raise() -> None:
    words: list[RecognizedWord] = []
    assign_reading_order(words)  # must not raise
    assert words == []


# ---- integration test (real model, slower) ----------------------------


def test_real_engine_recognizes_rendered_urdu_text() -> None:
    """Uses the same Nastaleeq font approach as tools/make_synthetic_test_pdf.py
    to render real Urdu text, then confirms PaddleOCREngine (real model,
    already cached from earlier sessions) returns at least one non-empty
    recognized word. This is a smoke test for the constructor/parsing
    logic against the real installed version, not an accuracy claim - CER/
    WER measurement is the Evaluation modules' job."""
    from pathlib import Path

    from PIL import Image, ImageDraw, ImageFont

    from app.core.recognition.paddle_ocr_engine import PaddleOCREngine

    font_path = Path(r"C:\Windows\Fonts\Jameel Noori Nastaleeq .ttf")
    if not font_path.exists():
        pytest.skip("Nastaleeq font not available on this machine")

    image = Image.new("L", (800, 200), color=255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 48)
    draw.text((400, 60), "\u0627\u0644\u062d\u0645\u062f \u0644\u0644\u0647", font=font, fill=0)  # "الحمد للہ"

    engine = PaddleOCREngine(lang="ur")
    words = engine.recognize(np.array(image))

    assert len(words) > 0
    assert all(w.text.strip() for w in words)
