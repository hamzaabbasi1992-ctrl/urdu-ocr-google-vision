"""Runs the minimal pipeline against the benchmark fixture and prints a
CER/WER/confidence/timing report - the actual deliverable of
PROJECT_SPEC.md Section 4's measurement-first build order.

This is a thin script, not an architecture module: it just calls
run_minimal_pipeline once and formats the result via BenchmarkReporter.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # run directly, not as a package -m

from app.core.evaluation.benchmark_reporter import BenchmarkReporter
from app.core.minimal_pipeline import run_minimal_pipeline
from app.core.recognition.paddle_ocr_engine import PaddleOCREngine

_FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"
_OUTPUT_DIR = Path(__file__).parent / "benchmark_output"


def main() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = PaddleOCREngine(lang="ur")
    result = run_minimal_pipeline(
        pdf_path=_FIXTURES / "benchmark_page.pdf",
        page_index=0,
        ground_truth_path=_FIXTURES / "benchmark_ground_truth.txt",
        output_txt_path=_OUTPUT_DIR / "output.txt",
        engine=engine,
    )

    report = BenchmarkReporter().format_report([result])
    print()
    print(report)

    report_path = _OUTPUT_DIR / "benchmark_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {report_path}")
    print(f"Recognized text written to {_OUTPUT_DIR / 'output.txt'}")


if __name__ == "__main__":
    main()
