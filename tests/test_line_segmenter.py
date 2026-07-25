"""Standalone test for Layer 5's LineSegmenter."""

from __future__ import annotations

import numpy as np

from app.core.structure.line_segmenter import LineSegmenter


def _image_with_lines(bands: list[tuple[int, int]], width: int = 300, height: int = 300) -> np.ndarray:
    image = np.full((height, width), 255, dtype=np.uint8)
    for y0, y1 in bands:
        image[y0:y1, 20:280] = 0  # solid ink band spanning most of the width
    return image


def test_detects_three_well_separated_lines() -> None:
    image = _image_with_lines([(20, 40), (80, 100), (140, 160)])
    regions = LineSegmenter().segment(image)

    assert len(regions) == 3
    assert regions[0].y0 < regions[1].y0 < regions[2].y0


def test_lines_are_ordered_top_to_bottom() -> None:
    image = _image_with_lines([(200, 220), (20, 40), (110, 130)])
    regions = LineSegmenter().segment(image)
    ys = [r.y0 for r in regions]
    assert ys == sorted(ys)


def test_small_internal_gap_stays_one_line() -> None:
    # Two bands with only a tiny gap between them (like a dip around a
    # diacritic-heavy word) - median line height ~20px, gap 4px < 0.35*20=7px
    image = _image_with_lines([(20, 40), (44, 60)])
    regions = LineSegmenter().segment(image)
    assert len(regions) == 1
    assert regions[0].y0 == 20
    assert regions[0].y1 == 60


def test_large_gap_stays_separate_lines() -> None:
    image = _image_with_lines([(20, 40), (100, 120)])  # gap of 60px, well beyond merge threshold
    regions = LineSegmenter().segment(image)
    assert len(regions) == 2


def test_blank_image_returns_no_lines() -> None:
    blank = np.full((300, 300), 255, dtype=np.uint8)
    assert LineSegmenter().segment(blank) == []


def test_crop_includes_padding_and_clamps_to_image_bounds() -> None:
    image = _image_with_lines([(20, 40)])
    segmenter = LineSegmenter()
    regions = segmenter.segment(image)
    cropped = segmenter.crop(image, regions[0], padding=4)

    assert cropped.shape[0] == (40 - 20) + 2 * 4  # padding on both sides
    assert cropped.shape[1] == image.shape[1]  # full width, unchanged


def test_crop_padding_clamps_at_image_edge() -> None:
    from app.core.structure.line_segmenter import LineRegion

    image = _image_with_lines([(0, 10)])
    segmenter = LineSegmenter()
    cropped = segmenter.crop(image, LineRegion(0, 10), padding=20)

    assert cropped.shape[0] <= image.shape[0]  # never exceeds the image, even with big padding
