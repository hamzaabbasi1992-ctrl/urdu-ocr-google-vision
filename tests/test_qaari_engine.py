"""Integration test for QaariVLMEngine - real model (base Qwen2-VL-2B +
LoRA adapter), no mocking. This model has no meaningful pure-logic surface
to unit test in isolation (unlike PaddleOCREngine's result-shape parsing) -
its entire value proposition is "does it actually recognize Nastaleeq
text," which can only be checked by running it. Slow: downloads/loads a
multi-GB base model on first run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_real_engine_recognizes_rendered_urdu_text() -> None:
    from PIL import Image, ImageDraw, ImageFont

    from app.core.recognition.qaari_engine import QaariVLMEngine

    font_path = Path(r"C:\Windows\Fonts\Jameel Noori Nastaleeq .ttf")
    if not font_path.exists():
        pytest.skip("Nastaleeq font not available on this machine")

    image = Image.new("L", (800, 200), color=255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 48)
    draw.text((400, 60), "\u0627\u0644\u062d\u0645\u062f \u0644\u0644\u0647", font=font, fill=0)  # "الحمد للہ"

    engine = QaariVLMEngine()
    words = engine.recognize(np.array(image))

    assert len(words) > 0
    assert all(w.text.strip() for w in words)
    assert all(0.0 <= w.confidence <= 1.0 for w in words)
