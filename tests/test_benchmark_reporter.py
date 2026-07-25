"""Standalone test for the Evaluation group's BenchmarkReporter - verifies
it formats results correctly and computes correct averages, without
computing any metric itself."""

from __future__ import annotations

from app.core.evaluation.benchmark_reporter import BenchmarkReporter, BenchmarkResult


def test_single_result_report_contains_all_fields() -> None:
    result = BenchmarkResult(
        label="page_1", cer=0.1234, wer=0.2345, average_confidence=0.876, processing_time_seconds=12.5
    )
    report = BenchmarkReporter().format_report([result])

    assert "page_1" in report
    assert "0.1234" in report
    assert "0.2345" in report
    assert "0.8760" in report
    assert "12.50" in report


def test_single_result_report_has_no_average_line() -> None:
    result = BenchmarkResult(label="page_1", cer=0.1, wer=0.1, average_confidence=0.9, processing_time_seconds=1.0)
    report = BenchmarkReporter().format_report([result])
    assert "Average" not in report


def test_multiple_results_report_includes_correct_average() -> None:
    results = [
        BenchmarkResult(label="page_1", cer=0.1, wer=0.2, average_confidence=0.8, processing_time_seconds=10.0),
        BenchmarkResult(label="page_2", cer=0.3, wer=0.4, average_confidence=0.6, processing_time_seconds=20.0),
    ]
    report = BenchmarkReporter().format_report(results)

    assert "page_1" in report
    assert "page_2" in report
    assert "Average: CER=0.2000  WER=0.3000  avg_confidence=0.7000  time=15.00s" in report


def test_empty_results_does_not_crash() -> None:
    report = BenchmarkReporter().format_report([])
    assert "Benchmark Report" in report
    assert "Average" not in report
