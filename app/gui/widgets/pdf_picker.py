"""PDF selection: add/remove files queued for batch processing."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QListWidget, QPushButton, QVBoxLayout, QWidget


class PdfPickerWidget(QWidget):
    files_changed = Signal(list)  # list[Path]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._files: list[Path] = []

        self.list_widget = QListWidget()

        add_button = QPushButton("Add PDF(s)...")
        add_button.clicked.connect(self.add_files_dialog)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._on_remove)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._on_clear)

        button_row = QHBoxLayout()
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)
        button_row.addWidget(clear_button)

        layout = QVBoxLayout(self)
        layout.addLayout(button_row)
        layout.addWidget(self.list_widget)

    def files(self) -> list[Path]:
        return list(self._files)

    def add_files_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select scanned PDF(s)", "", "PDF files (*.pdf)")
        for path_str in paths:
            path = Path(path_str)
            if path not in self._files:
                self._files.append(path)
                self.list_widget.addItem(path.name)
        self.files_changed.emit(self.files())

    def _on_remove(self) -> None:
        for item in self.list_widget.selectedItems():
            row = self.list_widget.row(item)
            self.list_widget.takeItem(row)
            del self._files[row]
        self.files_changed.emit(self.files())

    def _on_clear(self) -> None:
        self.list_widget.clear()
        self._files.clear()
        self.files_changed.emit(self.files())
