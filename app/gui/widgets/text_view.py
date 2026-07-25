"""Shows OCR text with per-word confidence highlighting. Read-only and never
edits recognized text - highlighting only marks low-confidence words for the
user to notice, it never substitutes or "fixes" anything.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget

from app.core.models import PageResult

_HIGH_CONF_COLOR = None  # no highlight
_MEDIUM_CONF_COLOR = QColor(255, 244, 190)  # pale yellow
_LOW_CONF_COLOR = QColor(255, 205, 205)  # pale red


class TextViewWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.summary_label = QLabel("No page loaded")

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setLayoutDirection(Qt.RightToLeft)

        layout = QVBoxLayout(self)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.text_edit)

    def clear(self) -> None:
        self.text_edit.clear()
        self.summary_label.setText("No page loaded")

    def set_page(self, page: PageResult) -> None:
        self.text_edit.clear()
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)

        lines: dict[int, list] = {}
        for word in page.words:
            lines.setdefault(word.line_index, []).append(word)

        for line_index in sorted(lines):
            for word in lines[line_index]:
                fmt = QTextCharFormat()
                if word.low_confidence:
                    fmt.setBackground(_LOW_CONF_COLOR)
                elif word.confidence < 0.9:
                    fmt.setBackground(_MEDIUM_CONF_COLOR)
                fmt.setToolTip(f"confidence: {word.confidence:.2f} ({word.engine})")
                cursor.insertText(word.text, fmt)
                cursor.insertText(" ", QTextCharFormat())
            cursor.insertBlock()

        low_conf_count = page.low_confidence_word_count
        self.summary_label.setText(
            f"Page {page.page_number} - {len(page.words)} word(s), "
            f"average confidence {page.average_confidence:.2f}, "
            f"{low_conf_count} flagged low-confidence"
        )
