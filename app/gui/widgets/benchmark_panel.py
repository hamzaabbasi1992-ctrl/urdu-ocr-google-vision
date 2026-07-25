"""Benchmark mode: runs a curated DPI/preprocessing grid against the current
page, shows average confidence per combination, and lets the user apply the
winning combination's settings to the main preprocessing panel.

User-triggered only, against one page at a time - never invoked
automatically during batch processing (see app/core/benchmark.py for why:
CPU-only OCR makes a full grid impractical to run on every page).
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.models import BenchmarkRunResult

_COLUMNS = ["Variant", "DPI", "Avg. Confidence", "Words", "Time (s)"]


class BenchmarkPanel(QWidget):
    run_clicked = Signal()
    apply_clicked = Signal(object)  # BenchmarkRunResult

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._results: list[BenchmarkRunResult] = []

        self.warning_label = QLabel(
            "Runs ~24 preprocessing/DPI combinations against the current page only. "
            "This machine has no GPU, so this can take several minutes."
        )
        self.warning_label.setWordWrap(True)

        self.run_button = QPushButton("Run Benchmark on Current Page")
        self.run_button.clicked.connect(self.run_clicked.emit)
        self.apply_button = QPushButton("Apply Selected Pipeline")
        self.apply_button.clicked.connect(self._on_apply)
        self.apply_button.setEnabled(False)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.itemSelectionChanged.connect(lambda: self.apply_button.setEnabled(bool(self._results)))

        button_row = QHBoxLayout()
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.apply_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.warning_label)
        layout.addLayout(button_row)
        layout.addWidget(self.table)

    def clear(self) -> None:
        self._results.clear()
        self.table.setRowCount(0)
        self.apply_button.setEnabled(False)

    def add_result(self, result: BenchmarkRunResult) -> None:
        self._results.append(result)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(result.variant_name))
        self.table.setItem(row, 1, QTableWidgetItem(str(result.dpi)))
        self.table.setItem(row, 2, QTableWidgetItem(f"{result.average_confidence:.3f}"))
        self.table.setItem(row, 3, QTableWidgetItem(str(result.word_count)))
        self.table.setItem(row, 4, QTableWidgetItem(f"{result.elapsed_seconds:.1f}"))

    def highlight_best(self) -> None:
        if not self._results:
            return
        best = max(self._results, key=lambda r: r.average_confidence)
        row = self._results.index(best)
        self.table.selectRow(row)
        self.apply_button.setEnabled(True)

    def _on_apply(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self._results):
            self.apply_clicked.emit(self._results[row])
