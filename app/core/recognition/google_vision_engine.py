"""Layer 6 - Recognition Engines: GoogleVisionEngine.

Single responsibility: recognize text via Google Cloud Vision's
DOCUMENT_TEXT_DETECTION only.

This is a CLOUD engine - per PROJECT_SPEC.md Section 2's revised offline
rule (explicit user sign-off obtained), pages processed through this
engine are sent to Google's servers. Its `name` and every place its output
is shown (GUI, CLI, reports) must keep that visible; it must never become
a silent default over the local engines.
"""

from __future__ import annotations

import numpy as np

from app.core.recognition.recognized_word import RecognizedWord, assign_reading_order


class GoogleVisionEngine:
    name = "google_vision_cloud"  # deliberately explicit that this is a cloud engine

    def __init__(self, credentials_path: str) -> None:
        import cv2  # noqa: F401 - imported here too so a missing cv2 fails at construction, not first use
        from google.cloud import vision
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(str(credentials_path))
        self._client = vision.ImageAnnotatorClient(credentials=credentials)
        self._vision = vision

    def recognize(self, image: np.ndarray) -> list[RecognizedWord]:
        import cv2

        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Could not encode image for Google Vision upload")

        vision_image = self._vision.Image(content=encoded.tobytes())
        response = self._client.document_text_detection(image=vision_image)

        if response.error.message:
            raise RuntimeError(f"Google Vision API error: {response.error.message}")

        words = parse_document_text_annotation(response.full_text_annotation)
        assign_reading_order(words)
        return words


def parse_document_text_annotation(annotation) -> list[RecognizedWord]:
    """Pulled out as a free function so it can be unit-tested against a
    constructed annotation object without a real API call."""
    words: list[RecognizedWord] = []
    for page in annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    text = "".join(symbol.text for symbol in word.symbols)
                    if not text.strip():
                        continue
                    xs = [v.x for v in word.bounding_box.vertices]
                    ys = [v.y for v in word.bounding_box.vertices]
                    words.append(
                        RecognizedWord(
                            text=text,
                            confidence=float(word.confidence),
                            x0=float(min(xs)),
                            y0=float(min(ys)),
                            x1=float(max(xs)),
                            y1=float(max(ys)),
                        )
                    )
    return words
