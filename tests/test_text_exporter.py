"""Standalone test for Layer 10's TextExporter - verifies line assembly
preserves given order (no reordering/rewriting) and that export writes
correctly-encoded content to disk."""

from __future__ import annotations

from pathlib import Path

from app.core.export.text_exporter import TextExporter, assemble_text
from app.core.recognition.paddle_ocr_engine import RecognizedWord


def _word(text: str, line_index: int) -> RecognizedWord:
    return RecognizedWord(text=text, confidence=1.0, x0=0, y0=0, x1=1, y1=1, line_index=line_index)


def test_assemble_text_joins_words_within_a_line_with_space() -> None:
    words = [_word("world", 0), _word("hello", 0)]  # already RTL-ordered by caller
    assert assemble_text(words) == "world hello"


def test_assemble_text_joins_lines_with_newline_in_line_index_order() -> None:
    words = [_word("line0word", 0), _word("line1word", 1), _word("line2word", 2)]
    assert assemble_text(words) == "line0word\nline1word\nline2word"


def test_assemble_text_does_not_reorder_words_within_a_line() -> None:
    """Words must come out in the exact order given, since ordering is
    PaddleOCREngine's responsibility, not this module's."""
    words = [_word("third", 0), _word("first", 0), _word("second", 0)]
    assert assemble_text(words) == "third first second"


def test_assemble_text_empty_list_returns_empty_string() -> None:
    assert assemble_text([]) == ""


def test_punctuation_attaches_directly_with_no_leading_space() -> None:
    words = [_word("ہے", 0), _word("۔", 0)]  # "ہے", "۔" (Urdu full stop)
    assert assemble_text(words) == "ہے۔"


def test_punctuation_after_multiple_words_still_attaches_to_the_last_one() -> None:
    words = [_word("world", 0), _word("hello", 0), _word(".", 0)]
    assert assemble_text(words) == "world hello."


def test_punctuation_as_the_very_first_token_does_not_crash() -> None:
    # No preceding word to attach to - must fall back to just appending it.
    words = [_word(".", 0), _word("word", 0)]
    assert assemble_text(words) == ". word"


def test_export_writes_utf8_sig_encoded_file(tmp_path: Path) -> None:
    path = tmp_path / "output.txt"
    TextExporter().export("الحمد لله", path)

    assert path.exists()
    assert path.read_text(encoding="utf-8-sig") == "الحمد لله"
    # Confirm a BOM is actually present (utf-8-sig, not plain utf-8)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_export_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "output.txt"
    TextExporter().export("text", path)
    assert path.exists()
