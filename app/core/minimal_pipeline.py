"""Thin orchestrator for the minimal pipeline (PROJECT_SPEC.md Section 4/
Q9): PDFLoader -> PageRasterizer -> Deskewer -> Denoiser ->
GlobalContrastEnhancer -> PaddleOCREngine -> TextExporter, then
GroundTruthLoader -> CERCalculator/WERCalculator/ConfidenceAggregator ->
BenchmarkResult.

Deliberately NOT a PageOrchestrator/DocumentOrchestrator/BatchController -
those imply job-queue/batch/pause-cancel concerns that are out of scope
for the minimal pipeline. This is one function, coordinating one page
through the chain once. It contains no business logic of its own - every
decision (how to deskew, how to denoise, how to score) belongs to the
module making it; this function only sequences calls and times the result.

Preprocessing order is denoise before contrast enhancement, not the
reverse - enhancing contrast first would amplify whatever noise the
denoiser was about to remove.

Render DPI is fixed (not adaptive), per Q9's explicit simplification:
adaptive DPI escalation (`MultiDPICandidateGenerator`/`SharpnessScorer`,
already built) is not part of the minimal pipeline until proven to
improve CER/WER over this fixed-DPI baseline.

DPI=200 (not the higher values tried initially) per a measured sweep
against the benchmark fixture: CER/WER showed no meaningful correlation
with render DPI at all (150-400 DPI all landed CER 0.80-0.87, WER stuck at
1.0), while processing time scaled from 7.7s to 29.7s - PaddleOCR's own
internal max_side_limit resize caps how much of the extra detail from a
higher-DPI render is even used. 200 DPI was chosen over the even-faster
150 DPI as a slightly safer margin given this is based on one synthetic
test page - see PROJECT_SPEC.md Section 7 for the full trade-off record.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.core.evaluation.benchmark_reporter import BenchmarkResult
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
from app.core.recognition.paddle_ocr_engine import PaddleOCREngine

RENDER_DPI = 200


def run_minimal_pipeline(
    pdf_path: Path,
    page_index: int,
    ground_truth_path: Path,
    output_txt_path: Path,
    engine: PaddleOCREngine,
) -> BenchmarkResult:
    start = time.monotonic()

    with PDFLoader(pdf_path) as loader:
        page = loader.get_page(page_index)
        rasterized = PageRasterizer().rasterize(page, dpi=RENDER_DPI)

    image = rasterized.image
    image = Deskewer().deskew(image).image
    image = Denoiser().denoise(image)
    image = GlobalContrastEnhancer().enhance(image)

    words = engine.recognize(image)
    text = assemble_text(words)
    TextExporter().export(text, output_txt_path)

    elapsed_seconds = time.monotonic() - start

    reference = GroundTruthLoader().load(ground_truth_path)
    cer = CERCalculator().calculate(text, reference)
    wer = WERCalculator().calculate(text, reference)
    average_confidence = ConfidenceAggregator().average(w.confidence for w in words)

    return BenchmarkResult(
        label=f"{pdf_path.name}#{page_index}",
        cer=cer,
        wer=wer,
        average_confidence=average_confidence,
        processing_time_seconds=elapsed_seconds,
    )
