"""Builds the benchmark ground-truth fixture: a single-page synthetic
"low-quality scan" PDF plus the exact reference text it was rendered from.

Reuses the font-rendering/degradation approach from
make_synthetic_test_pdf.py (same Nastaleeq font, same shadow/noise/blur/
downscale-upscale degradation) rather than re-deriving it - see
PROJECT_SPEC.md Section 4/7 on not rewriting things that already work.

The ground-truth text is the exact source lines, joined the same way
TextExporter.assemble_text joins recognized lines (newline-separated,
logical Unicode reading order - RTL rendering is a display concern, not a
storage-order concern), so CER/WER comparisons are apples-to-apples.

This is still synthetic, not a real scan - per the standing
iterate-from-real-usage principle, real scanned pages (once available)
are the actual accuracy test; this fixture only proves the pipeline and
benchmark plumbing work end-to-end.
"""

from __future__ import annotations

from pathlib import Path

from make_synthetic_test_pdf import _SAMPLE_LINES, _clean_page, _degrade


def build_benchmark_fixture(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    clean = _clean_page(width=1700, height=2200, font_size=64)
    degraded = _degrade(clean, seed=0).convert("RGB")

    pdf_path = output_dir / "benchmark_page.pdf"
    degraded.save(pdf_path, resolution=200.0)

    ground_truth_path = output_dir / "benchmark_ground_truth.txt"
    ground_truth_path.write_text("\n".join(_SAMPLE_LINES), encoding="utf-8-sig")

    print(f"Wrote {pdf_path}")
    print(f"Wrote {ground_truth_path}")


if __name__ == "__main__":
    build_benchmark_fixture(Path(__file__).parent.parent / "tests" / "fixtures")
