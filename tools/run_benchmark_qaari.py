"""Runs Qaari against the same benchmark fixture PaddleOCR was measured
against, for a direct comparison. Mirrors run_benchmark.py but for
QaariVLMEngine instead of PaddleOCREngine - kept as a separate thin script
rather than generalizing run_benchmark.py to take an engine argument,
since this is a one-off comparison run, not a permanent CLI surface yet.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.evaluation.benchmark_reporter import BenchmarkReporter, BenchmarkResult
from app.core.evaluation.cer_calculator import CERCalculator
from app.core.evaluation.confidence_aggregator import ConfidenceAggregator
from app.core.evaluation.ground_truth_loader import GroundTruthLoader
from app.core.evaluation.wer_calculator import WERCalculator
from app.core.export.text_exporter import TextExporter, assemble_text
from app.core.ingestion.pdf_loader import PDFLoader
from app.core.ingestion.page_rasterizer import PageRasterizer
from app.core.preprocessing.contrast_enhancer import GlobalContrastEnhancer
from app.core.preprocessing.denoiser import Denoiser
from app.core.preprocessing.deskewer import Deskewer
from app.core.recognition.qaari_engine import QaariVLMEngine

_FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"
_OUTPUT_DIR = Path(__file__).parent / "benchmark_output"


def main() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    engine = QaariVLMEngine()

    with PDFLoader(_FIXTURES / "benchmark_page.pdf") as loader:
        page = loader.get_page(0)
        rasterized = PageRasterizer().rasterize(page, dpi=200)

    image = rasterized.image
    image = Deskewer().deskew(image).image
    image = Denoiser().denoise(image)
    image = GlobalContrastEnhancer().enhance(image)

    words = engine.recognize(image)
    text = assemble_text(words)
    elapsed = time.monotonic() - start

    output_path = _OUTPUT_DIR / "output_qaari.txt"
    TextExporter().export(text, output_path)

    reference = GroundTruthLoader().load(_FIXTURES / "benchmark_ground_truth.txt")
    cer = CERCalculator().calculate(text, reference)
    wer = WERCalculator().calculate(text, reference)
    confidence = ConfidenceAggregator().average(w.confidence for w in words)

    result = BenchmarkResult(
        label="benchmark_page.pdf#0 (Qaari)", cer=cer, wer=wer,
        average_confidence=confidence, processing_time_seconds=elapsed,
    )
    print()
    print(BenchmarkReporter().format_report([result]))
    print(f"\nRecognized text written to {output_path}")
    print("\n--- Recognized text ---")
    print(text)


if __name__ == "__main__":
    main()
