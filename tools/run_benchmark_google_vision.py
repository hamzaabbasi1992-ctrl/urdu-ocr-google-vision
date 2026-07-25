"""Runs Google Vision against the same benchmark fixture PaddleOCR/Qaari
were measured against. Requires GOOGLE_VISION_CREDENTIALS_PATH to be set.
"""

from __future__ import annotations

import os
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
from app.core.recognition.google_vision_engine import GoogleVisionEngine

_FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"
_OUTPUT_DIR = Path(__file__).parent / "benchmark_output"


def main() -> None:
    credentials_path = os.environ.get("GOOGLE_VISION_CREDENTIALS_PATH")
    if not credentials_path:
        raise SystemExit("Set GOOGLE_VISION_CREDENTIALS_PATH to the service account JSON key path")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    engine = GoogleVisionEngine(credentials_path=credentials_path)

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

    output_path = _OUTPUT_DIR / "output_google_vision.txt"
    TextExporter().export(text, output_path)

    reference = GroundTruthLoader().load(_FIXTURES / "benchmark_ground_truth.txt")
    cer = CERCalculator().calculate(text, reference)
    wer = WERCalculator().calculate(text, reference)
    confidence = ConfidenceAggregator().average(w.confidence for w in words)

    result = BenchmarkResult(
        label="benchmark_page.pdf#0 (Google Vision - CLOUD)", cer=cer, wer=wer,
        average_confidence=confidence, processing_time_seconds=elapsed,
    )
    report = BenchmarkReporter().format_report([result])
    sys.stdout.buffer.write(("\n" + report + f"\nRecognized text written to {output_path}\n").encode("utf-8"))
    sys.stdout.buffer.write(("\n--- Recognized text ---\n" + text + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
