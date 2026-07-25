"""Sequential batch runner: pause/resume/cancel, per-page progress, output
writing.

Sequential by design - PaddleOCR/Tesseract run CPU-only on this machine (no
GPU), so pages are processed one at a time against cached engine instances
rather than in parallel across pages/files; any concurrency belongs inside
a single page's OCR call, not across pages.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import cv2

from app.core.control import Cancelled, JobControl
from app.core.exporters.docx_exporter import export_docx
from app.core.exporters.json_exporter import export_json
from app.core.exporters.searchable_pdf_exporter import export_searchable_pdf
from app.core.exporters.txt_exporter import export_txt
from app.core.job import JobStatus, OCRJob
from app.core.model_manager import ModelManager
from app.core.models import DocumentResult, OCRConfig, PageResult, PreprocessConfig
from app.core.ocr.arbiter import Arbiter
from app.core.ocr.multiscale import run_multiscale
from app.core.pdf_render import (
    cleanup_document_temp_dir,
    cleanup_page_render,
    new_document_temp_dir,
    open_pdf,
    page_count,
    render_page_adaptive,
)
from app.core.postprocess import postprocess_page
from app.core.preprocess.pipeline import run_pipeline

_LOGGER = logging.getLogger("urdu_ocr.job_queue")

JobCallback = Callable[[OCRJob], None]
PageCallback = Callable[[OCRJob, PageResult], None]
StageCallback = Callable[[OCRJob, str, bool], None]


class JobQueue:
    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    def run(
        self,
        jobs: list[OCRJob],
        preprocess_config: PreprocessConfig,
        ocr_config: OCRConfig,
        output_root: Path,
        control: JobControl,
        batch_logger: logging.Logger,
        on_job_update: JobCallback | None = None,
        on_page: PageCallback | None = None,
        on_stage: StageCallback | None = None,
    ) -> None:
        control.reset()
        arbiter = self._model_manager.get_arbiter(ocr_config)

        for job in jobs:
            if control.is_cancelled():
                job.status = JobStatus.CANCELLED
                if on_job_update:
                    on_job_update(job)
                continue

            job.status = JobStatus.RUNNING
            job.started_at = time.monotonic()
            if on_job_update:
                on_job_update(job)
            batch_logger.info("Starting %s", job.source_path)

            try:
                job.total_pages = page_count(job.source_path)
                document_result = DocumentResult(source_path=job.source_path)
                job.result = document_result

                output_folder = output_root / job.source_path.stem
                pages_dir = output_folder / "pages"
                pages_dir.mkdir(parents=True, exist_ok=True)

                temp_dir = new_document_temp_dir(job.source_path)
                doc = open_pdf(job.source_path)
                try:
                    for page_index in range(job.total_pages):
                        control.checkpoint()
                        if on_stage:
                            on_stage(job, f"Processing page {page_index + 1}/{job.total_pages}", False)

                        page_result = self._process_page(
                            doc, page_index, preprocess_config, ocr_config, arbiter, temp_dir, pages_dir
                        )
                        document_result.pages.append(page_result)
                        job.processed_pages += 1
                        if on_page:
                            on_page(job, page_result)
                        if on_job_update:
                            on_job_update(job)
                finally:
                    doc.close()
                    cleanup_document_temp_dir(temp_dir)

                if on_stage:
                    on_stage(job, "Writing outputs", True)
                self._write_outputs(document_result, output_folder)
                job.output_folder = output_folder
                job.status = JobStatus.DONE
                job.finished_at = time.monotonic()
                batch_logger.info(
                    "Finished %s (%d pages) -> %s", job.source_path, job.total_pages, job.output_folder
                )
            except Cancelled:
                job.status = JobStatus.CANCELLED
                job.finished_at = time.monotonic()
                batch_logger.warning("Cancelled %s", job.source_path)
            except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
                job.status = JobStatus.ERROR
                job.error_message = str(exc)
                job.finished_at = time.monotonic()
                batch_logger.error("Error processing %s: %s", job.source_path, exc, exc_info=True)

            if on_job_update:
                on_job_update(job)

    def _process_page(
        self,
        doc,
        page_index: int,
        preprocess_config: PreprocessConfig,
        ocr_config: OCRConfig,
        arbiter: Arbiter,
        temp_dir: Path,
        pages_dir: Path,
    ) -> PageResult:
        adaptive = render_page_adaptive(doc, page_index, ocr_config, temp_dir)
        chosen_dpi = adaptive.chosen.dpi

        processed, debug = run_pipeline(adaptive.chosen.image, preprocess_config, chosen_dpi)

        if ocr_config.multiscale_ocr and len(adaptive.tried_dpis) > 1:
            images_by_dpi = {chosen_dpi: processed}
            for render in adaptive.discarded:
                variant_processed, _ = run_pipeline(render.image, preprocess_config, render.dpi)
                images_by_dpi[render.dpi] = variant_processed
            words = run_multiscale(images_by_dpi, reference_dpi=chosen_dpi, arbiter=arbiter)
        else:
            words = arbiter.run(processed)

        canonical_image = debug.canonical_image if debug.canonical_image is not None else adaptive.chosen.image
        for word in words:
            word.bbox = debug.transform.to_canonical(word.bbox)

        raw_path = pages_dir / f"page{page_index + 1:04d}_raw.png"
        cv2.imwrite(str(raw_path), canonical_image, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        text = postprocess_page(words)

        page_result = PageResult(
            page_number=page_index + 1,
            width=int(canonical_image.shape[1]),
            height=int(canonical_image.shape[0]),
            words=words,
            chosen_dpi=chosen_dpi,
            sharpness_score=adaptive.sharpness_score,
            orientation_angle=debug.orientation_angle,
            raw_image_path=raw_path,
            text=text,
        )

        cleanup_page_render(adaptive, keep_chosen=False)  # canonical copy already persisted above
        return page_result

    def _write_outputs(self, result: DocumentResult, output_folder: Path) -> None:
        output_folder.mkdir(parents=True, exist_ok=True)
        export_txt(result, output_folder / "output.txt")
        export_docx(result, output_folder / "output.docx")
        export_searchable_pdf(result, output_folder / "output_searchable.pdf")
        export_json(result, output_folder / "output.json")
