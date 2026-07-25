"""output.txt - the literal OCR text, one blank line between pages.

Written with a UTF-8 BOM (utf-8-sig) so Windows Notepad renders Urdu
correctly by default rather than guessing the wrong codepage.
"""

from __future__ import annotations

from pathlib import Path

from app.core.models import DocumentResult


def export_txt(result: DocumentResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write(result.full_text)
