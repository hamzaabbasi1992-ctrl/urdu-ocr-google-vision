"""Layer 1 - Ingestion: PDFLoader.

Single responsibility: open a PDF file and expose its page count and basic
metadata. Nothing else - it does not rasterize pages (see the upcoming
PageRasterizer module) and does not judge anything about page content or
quality. Per PROJECT_SPEC.md Section 4, a module that opens a file does not
also decide what to do with what's inside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass(slots=True, frozen=True)
class PDFMetadata:
    page_count: int
    title: str | None
    author: str | None


class PDFLoader:
    """Opens a PDF and exposes page count/metadata/raw pages. Does not render."""

    def __init__(self, pdf_path: Path) -> None:
        self._pdf_path = Path(pdf_path)
        if not self._pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self._pdf_path}")
        self._doc = fitz.open(self._pdf_path)

    @property
    def path(self) -> Path:
        return self._pdf_path

    @property
    def page_count(self) -> int:
        return self._doc.page_count

    def metadata(self) -> PDFMetadata:
        meta = self._doc.metadata or {}
        return PDFMetadata(
            page_count=self._doc.page_count,
            title=meta.get("title") or None,
            author=meta.get("author") or None,
        )

    def get_page(self, index: int) -> fitz.Page:
        """Returns the underlying fitz.Page for a 0-based index - the only
        thing PageRasterizer needs from this module. PDFLoader does not
        interpret, render, or otherwise judge the page itself."""
        if not 0 <= index < self.page_count:
            raise IndexError(f"Page {index} out of range (0..{self.page_count - 1})")
        return self._doc[index]

    def close(self) -> None:
        self._doc.close()

    def __enter__(self) -> "PDFLoader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
