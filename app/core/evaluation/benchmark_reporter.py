"""Evaluation & Benchmarking: BenchmarkReporter.

Single responsibility: given CER/WER/confidence/timing results, produce a
report. Formatting only - does not compute any metric itself (that's
CERCalculator/WERCalculator/ConfidenceAggregator's job).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class BenchmarkResult:
    label: str
    cer: float
    wer: float
    average_confidence: float
    processing_time_seconds: float


class BenchmarkReporter:
    def format_report(self, results: list[BenchmarkResult]) -> str:
        lines = ["Benchmark Report", "=" * 60]
        for result in results:
            lines.append(
                f"{result.label}: CER={result.cer:.4f}  WER={result.wer:.4f}  "
                f"avg_confidence={result.average_confidence:.4f}  "
                f"time={result.processing_time_seconds:.2f}s"
            )

        if len(results) > 1:
            lines.append("-" * 60)
            lines.append(
                f"Average: CER={self._avg(results, 'cer'):.4f}  "
                f"WER={self._avg(results, 'wer'):.4f}  "
                f"avg_confidence={self._avg(results, 'average_confidence'):.4f}  "
                f"time={self._avg(results, 'processing_time_seconds'):.2f}s"
            )

        return "\n".join(lines)

    @staticmethod
    def _avg(results: list[BenchmarkResult], field: str) -> float:
        values = [getattr(r, field) for r in results]
        return sum(values) / len(values)
