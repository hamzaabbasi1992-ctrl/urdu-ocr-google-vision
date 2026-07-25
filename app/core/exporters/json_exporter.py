"""JSON export: page number, word, bounding box, confidence for every word."""

from __future__ import annotations

from pathlib import Path

import orjson

from app.core.models import DocumentResult


def export_json(result: DocumentResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pages_payload = []
    for page in result.pages:
        words_payload = [
            {
                "word": word.text,
                "bbox": list(word.bbox.as_tuple()),
                "confidence": round(word.confidence, 4),
                "engine": word.engine,
                "low_confidence": word.low_confidence,
                "is_diacritic": word.is_diacritic,
            }
            for word in page.words
        ]
        pages_payload.append(
            {
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "chosen_dpi": page.chosen_dpi,
                "average_confidence": round(page.average_confidence, 4),
                "words": words_payload,
            }
        )

    payload = {"source": str(result.source_path), "pages": pages_payload}
    output_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))
