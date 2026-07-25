"""Standalone test for Layer 6's GoogleVisionEngine.

Split into pure-logic tests against constructed (real protobuf-typed, not
hand-mocked) Vision API response objects - no network call - and one real
API integration test gated behind the actual service account credentials
file, which is not committed to the repo (see .gitignore) and whose path
must be provided via the GOOGLE_VISION_CREDENTIALS_PATH env var to run.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from app.core.recognition.google_vision_engine import parse_document_text_annotation

# ---- pure-logic tests (no network) ------------------------------------


def _make_word(text: str, confidence: float, x0: int, y0: int, x1: int, y1: int):
    from google.cloud import vision

    return vision.Word(
        symbols=[vision.Symbol(text=ch) for ch in text],
        bounding_box=vision.BoundingPoly(
            vertices=[
                vision.Vertex(x=x0, y=y0),
                vision.Vertex(x=x1, y=y0),
                vision.Vertex(x=x1, y=y1),
                vision.Vertex(x=x0, y=y1),
            ]
        ),
        confidence=confidence,
    )


def _annotation_with_words(*words):
    from google.cloud import vision

    paragraph = vision.Paragraph(words=list(words))
    block = vision.Block(paragraphs=[paragraph])
    page = vision.Page(blocks=[block])
    return vision.TextAnnotation(pages=[page])


def test_parses_word_text_from_symbols() -> None:
    annotation = _annotation_with_words(_make_word("\u0627\u0628", 0.9, 10, 10, 50, 30))
    words = parse_document_text_annotation(annotation)

    assert len(words) == 1
    assert words[0].text == "\u0627\u0628"
    assert words[0].confidence == pytest.approx(0.9, abs=1e-4)


def test_bounding_box_computed_from_vertices() -> None:
    annotation = _annotation_with_words(_make_word("x", 0.5, 10, 20, 50, 80))
    words = parse_document_text_annotation(annotation)

    assert (words[0].x0, words[0].y0, words[0].x1, words[0].y1) == (10.0, 20.0, 50.0, 80.0)


def test_skips_blank_words() -> None:
    from google.cloud import vision

    blank = vision.Word(
        symbols=[vision.Symbol(text=" ")],
        bounding_box=vision.BoundingPoly(vertices=[vision.Vertex(x=0, y=0)] * 4),
        confidence=0.9,
    )
    annotation = _annotation_with_words(blank, _make_word("real", 0.8, 0, 0, 10, 10))

    words = parse_document_text_annotation(annotation)
    assert len(words) == 1
    assert words[0].text == "real"


def test_empty_annotation_returns_no_words() -> None:
    from google.cloud import vision

    annotation = vision.TextAnnotation(pages=[])
    assert parse_document_text_annotation(annotation) == []


def test_multiple_words_across_blocks_and_paragraphs() -> None:
    from google.cloud import vision

    p1 = vision.Paragraph(words=[_make_word("one", 0.9, 0, 0, 10, 10)])
    p2 = vision.Paragraph(words=[_make_word("two", 0.8, 20, 0, 30, 10)])
    block1 = vision.Block(paragraphs=[p1, p2])
    block2 = vision.Block(paragraphs=[vision.Paragraph(words=[_make_word("three", 0.7, 0, 50, 10, 60)])])
    annotation = vision.TextAnnotation(pages=[vision.Page(blocks=[block1, block2])])

    words = parse_document_text_annotation(annotation)
    assert {w.text for w in words} == {"one", "two", "three"}


# ---- real API integration test -----------------------------------------


def test_real_api_recognizes_rendered_urdu_text() -> None:
    credentials_path = os.environ.get("GOOGLE_VISION_CREDENTIALS_PATH")
    if not credentials_path or not Path(credentials_path).exists():
        pytest.skip("GOOGLE_VISION_CREDENTIALS_PATH not set - skipping real API call")

    from PIL import Image, ImageDraw, ImageFont

    from app.core.recognition.google_vision_engine import GoogleVisionEngine

    font_path = Path(r"C:\Windows\Fonts\Jameel Noori Nastaleeq .ttf")
    if not font_path.exists():
        pytest.skip("Nastaleeq font not available on this machine")

    image = Image.new("L", (800, 200), color=255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 48)
    draw.text((400, 60), "\u0627\u0644\u062d\u0645\u062f \u0644\u0644\u0647", font=font, fill=0)  # "الحمد للہ"

    engine = GoogleVisionEngine(credentials_path=credentials_path)
    words = engine.recognize(np.array(image))

    assert len(words) > 0
    assert all(w.text.strip() for w in words)
