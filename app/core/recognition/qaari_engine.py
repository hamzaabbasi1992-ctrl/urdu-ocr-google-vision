"""Layer 6 - Recognition Engines: QaariVLMEngine.

Single responsibility: recognize text via Qaari-0.1 (a Qwen2-VL-2B model
fine-tuned, via a LoRA adapter, specifically on Nastaleeq/Suls fonts - see
PROJECT_SPEC.md Section 6 for why this is being measured now rather than
assumed to help).

Unlike PaddleOCR, this is a line-recognition model, not a page-level
detector+recognizer - it takes one cropped text line image and generates
text for it. LineSegmenter (Layer 5) provides the line crops; this module
does not do its own detection.

It also has no native confidence score (it's a generation model, not a
classifier), so confidence is approximated from the mean per-token
generation probability - a documented approximation, not a calibrated
score comparable to PaddleOCR's.

The distributed weights are a ~96MB LoRA adapter over
"Qwen/Qwen2-VL-2B-Instruct". The adapter card's example loads a 4-bit
bitsandbytes build of the base model, which needs a CUDA GPU; there is no
GPU on this machine, so this loads the full-precision base model and
applies the adapter with peft, running on CPU.
"""

from __future__ import annotations

import logging

import numpy as np

from app.core.paths import model_cache_dir
from app.core.recognition.paddle_ocr_engine import RecognizedWord, assign_reading_order
from app.core.structure.line_segmenter import LineSegmenter

_LOGGER = logging.getLogger("urdu_ocr.recognition.qaari_engine")

_BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
_ADAPTER_MODEL_ID = "oddadmix/Qaari-0.1-Urdu-OCR-VL-2B-Instruct"
_PROMPT = (
    "Below is the image of one page of a document, as well as some raw "
    "textual content that was previously extracted for it. Just return "
    "the plain text representation of this document as if you were "
    "reading it naturally. Do not hallucinate."
)


class QaariVLMEngine:
    name = "qaari"

    def __init__(self, max_new_tokens: int = 128) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self._torch = torch
        self._max_new_tokens = max_new_tokens
        cache_dir = str(model_cache_dir() / "huggingface")

        _LOGGER.info("Loading Qaari (Qwen2-VL-2B base + LoRA adapter) on CPU - one-time, may take a while")
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            _BASE_MODEL_ID, torch_dtype=torch.float32, cache_dir=cache_dir
        )
        self._model = PeftModel.from_pretrained(base_model, _ADAPTER_MODEL_ID, cache_dir=cache_dir)
        self._model.to("cpu")
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(_BASE_MODEL_ID, cache_dir=cache_dir)
        self._segmenter = LineSegmenter()

    def recognize(self, image: np.ndarray) -> list[RecognizedWord]:
        regions = self._segmenter.segment(image)
        words: list[RecognizedWord] = []

        for line_index, region in enumerate(regions):
            line_image = self._segmenter.crop(image, region)
            text, confidence = self._recognize_line(line_image)
            if not text.strip():
                continue
            for token in text.split():
                words.append(
                    RecognizedWord(
                        text=token,
                        confidence=confidence,
                        x0=0.0,
                        y0=float(region.y0),
                        x1=float(image.shape[1]),
                        y1=float(region.y1),
                        line_index=line_index,
                    )
                )

        assign_reading_order(words)
        return words

    def _recognize_line(self, line_image: np.ndarray) -> tuple[str, float]:
        from PIL import Image

        torch = self._torch
        pil_image = Image.fromarray(line_image).convert("RGB")

        messages = [
            {"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": _PROMPT}]}
        ]
        chat_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[chat_text], images=[pil_image], padding=True, return_tensors="pt")

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output.sequences[0][input_length:]
        text = self._processor.decode(generated_ids, skip_special_tokens=True).strip()
        confidence = _mean_token_confidence(output.scores, generated_ids, torch)
        return text, confidence


def _mean_token_confidence(scores, generated_ids, torch) -> float:
    """Average probability the model assigned to each token it actually
    generated - an approximation, not a calibrated confidence, since
    generation models have no native word-level confidence output."""
    if not scores:
        return 0.0
    probs = []
    for step_logits, token_id in zip(scores, generated_ids):
        step_probs = torch.softmax(step_logits[0], dim=-1)
        probs.append(step_probs[token_id].item())
    return float(sum(probs) / len(probs)) if probs else 0.0
