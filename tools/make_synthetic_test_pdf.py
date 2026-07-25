"""Builds a synthetic "low-quality scan" PDF for end-to-end pipeline
verification, since no real scanned samples are available yet (see the
plan's Verification section). Renders Urdu text with a real Windows Urdu
font, then degrades it (blur, rotation, noise, shadow gradient) to roughly
approximate a cheap scan/photograph of a book page.

This is only good for confirming the plumbing works and every
preprocessing toggle visibly does something - it is not a substitute for
validating OCR accuracy against real scans.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

_CANDIDATE_FONTS = [
    r"C:\Windows\Fonts\Jameel Noori Nastaleeq .ttf",  # real Nastaleeq font found on this machine
    r"C:\Windows\Fonts\Alvi Nastaleeq.ttf",
    r"C:\Windows\Fonts\Nafees Nastaleeq v1.02(VOLT project).ttf",
    r"C:\Windows\Fonts\UrduTypesetting.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]

_SAMPLE_LINES = [
    "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
    "یہ ایک آزمائشی صفحہ ہے جو اردو نستعلیق رسم الخط میں لکھا گیا ہے۔",
    "علم حاصل کرو چاہے تمہیں چین ہی کیوں نہ جانا پڑے۔",
    "الحمد للہ رب العالمین، والصلاة والسلام على سيد المرسلين۔",
    "نمبر: ۱۲۳۴۵۶۷۸۹۰ - یہ ہندسے بھی محفوظ رہنے چاہئیں۔",
]


def _find_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _CANDIDATE_FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError("No suitable Urdu-capable font found on this system.")


def _clean_page(width: int, height: int, font_size: int) -> Image.Image:
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    font = _find_font(font_size)

    y = height // 6
    for line in _SAMPLE_LINES:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = width - text_width - width // 8  # right-aligned, RTL page
        draw.text((x, y), line, font=font, fill=0)
        y += font_size + int(font_size * 0.8)

    return image


def _degrade(image: Image.Image, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.array(image, dtype=np.float32)
    h, w = arr.shape

    # Shadow gradient across the page
    gradient = np.linspace(0.75, 1.0, w, dtype=np.float32)
    arr *= gradient[np.newaxis, :]

    # Gaussian noise
    arr += rng.normal(0, 8, size=arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    degraded = Image.fromarray(arr, mode="L")
    degraded = degraded.rotate(0.7, resample=Image.BICUBIC, fillcolor=255)
    degraded = degraded.filter(ImageFilter.GaussianBlur(radius=1.1))

    # Downscale then upscale to lose detail, like a low-resolution source scan
    small = degraded.resize((w // 3, h // 3), Image.BILINEAR)
    degraded = small.resize((w, h), Image.BILINEAR)

    return degraded


def build_synthetic_pdf(output_path: Path, pages: int = 2) -> None:
    images = []
    for i in range(pages):
        clean = _clean_page(width=1700, height=2200, font_size=64)
        degraded = _degrade(clean, seed=i)
        images.append(degraded.convert("RGB"))

    # Pillow assumes 72 DPI for PDF page-size math unless told otherwise; at
    # 1700x2200px that would make the "page" ~24x31 real-world inches, which
    # then renders to a 100+ megapixel image at normal OCR DPIs - explicit
    # resolution keeps this a normal ~8.5x11in page.
    images[0].save(output_path, save_all=True, append_images=images[1:], resolution=200.0)
    print(f"Wrote {output_path} ({pages} page(s))")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "synthetic_test.pdf"
    build_synthetic_pdf(out)
