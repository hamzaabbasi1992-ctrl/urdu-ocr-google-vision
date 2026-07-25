"""Regression test for a real bug found during integration testing: pinning
`text_detection_model_name` (needed to work around a native crash - see
PaddleOCREngine's docstring) silently drops PaddleOCR's `lang`-based
recognition-model auto-selection too, unless `text_recognition_model_name`
is pinned in the same call. An earlier version of this fix pinned only the
detection model, which produced a passing test (no crash) but garbled,
wrong-language output (CER 0.98) - a green test that was actually wrong.
This test cannot catch the deeper PaddleOCR behavior without loading real
models (see test_paddle_ocr_engine.py's integration test for that), but it
does pin the language->model mapping table itself against silent edits.
"""

from __future__ import annotations

from app.core.recognition.paddle_ocr_engine import _recognition_model_for_lang


def test_urdu_maps_to_arabic_recognition_model() -> None:
    assert _recognition_model_for_lang("ur") == "arabic_PP-OCRv5_mobile_rec"


def test_unmapped_language_returns_none_rather_than_guessing() -> None:
    assert _recognition_model_for_lang("zz-not-a-real-lang") is None
