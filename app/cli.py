"""Headless batch entry point - reuses the same JobQueue pipeline as the
GUI, for scripted/unattended runs and for automated verification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core.control import JobControl
from app.core.job import JobStatus, OCRJob
from app.core.job_queue import JobQueue
from app.core.logging_setup import create_batch_log, setup_app_logging
from app.core.model_manager import ModelManager
from app.core.models import OCRConfig, PreprocessConfig
from app.core.paths import default_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline OCR for scanned Urdu Nastaleeq PDFs")
    parser.add_argument("pdfs", nargs="+", type=Path, help="One or more PDF files to OCR")
    parser.add_argument("--output", type=Path, default=default_output_dir(), help="Output folder")
    parser.add_argument("--no-tesseract-fallback", action="store_true")
    parser.add_argument("--no-adaptive-dpi", action="store_true")
    args = parser.parse_args()

    setup_app_logging()
    batch_logger, log_path = create_batch_log(args.output)

    jobs = [OCRJob(source_path=p) for p in args.pdfs]
    preprocess_config = PreprocessConfig()
    ocr_config = OCRConfig(
        use_tesseract_fallback=not args.no_tesseract_fallback,
        adaptive_dpi=not args.no_adaptive_dpi,
    )

    queue = JobQueue(ModelManager())
    control = JobControl()

    def on_job_update(job: OCRJob) -> None:
        print(f"[{job.status.value}] {job.source_path.name} ({job.processed_pages}/{job.total_pages or '?'})")

    queue.run(jobs, preprocess_config, ocr_config, args.output, control, batch_logger, on_job_update=on_job_update)

    failed = [j for j in jobs if j.status != JobStatus.DONE]
    print(f"\nLog: {log_path}")
    if failed:
        for job in failed:
            print(f"FAILED: {job.source_path} - {job.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
