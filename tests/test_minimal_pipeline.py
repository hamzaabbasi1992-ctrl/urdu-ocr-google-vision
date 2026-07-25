"""Integration test for the minimal-pipeline orchestrator.

This is deliberately a real end-to-end test, not a mocked/stubbed one -
the orchestrator's only real value is proving the actual wiring between
real modules works, so stubbing every dependency would just prove that
Python function calls happen in the order they're written, which is not
worth much. It reuses the already-cached PaddleOCR model (no network
needed) and the benchmark_page.pdf/benchmark_ground_truth.txt fixture
(tools/make_benchmark_fixture.py). Slow (~model load + inference time),
matching test_paddle_ocr_engine.py's real-model integration test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.minimal_pipeline import run_minimal_pipeline
from app.core.recognition.paddle_ocr_engine import PaddleOCREngine

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def paddle_engine() -> PaddleOCREngine:
    return PaddleOCREngine(lang="ur")


def test_minimal_pipeline_end_to_end(tmp_path: Path, paddle_engine: PaddleOCREngine) -> None:
    pdf_path = _FIXTURES / "benchmark_page.pdf"
    ground_truth_path = _FIXTURES / "benchmark_ground_truth.txt"
    if not pdf_path.exists() or not ground_truth_path.exists():
        pytest.skip("Benchmark fixture not generated - run tools/make_benchmark_fixture.py first")

    output_txt_path = tmp_path / "output.txt"

    result = run_minimal_pipeline(
        pdf_path=pdf_path,
        page_index=0,
        ground_truth_path=ground_truth_path,
        output_txt_path=output_txt_path,
        engine=paddle_engine,
    )

    assert output_txt_path.exists()
    assert output_txt_path.read_text(encoding="utf-8-sig").strip() != ""

    assert 0.0 <= result.cer <= 5.0  # CER can exceed 1.0 with enough insertions; must at least be non-negative
    assert 0.0 <= result.wer <= 5.0
    assert 0.0 <= result.average_confidence <= 1.0
    assert result.processing_time_seconds > 0.0
    assert result.label == "benchmark_page.pdf#0"
