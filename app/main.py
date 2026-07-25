"""GUI entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.core.logging_setup import setup_app_logging


def main() -> None:
    setup_app_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("Urdu OCR")

    from app.gui.main_window import MainWindow

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
