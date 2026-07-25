"""Light and dark Qt stylesheets, toggleable at runtime (GUI spec: dark mode)."""

from __future__ import annotations

DARK_STYLESHEET = """
QWidget {
    background-color: #1e1f22;
    color: #e6e6e6;
    font-family: 'Segoe UI', sans-serif;
    font-size: 10.5pt;
}
QMainWindow { background-color: #1e1f22; }
QPushButton {
    background-color: #2b2d31;
    border: 1px solid #3f4147;
    border-radius: 4px;
    padding: 6px 14px;
}
QPushButton:hover { background-color: #35373c; }
QPushButton:pressed { background-color: #26282c; }
QPushButton:disabled { color: #6b6d70; background-color: #232427; border-color: #2c2d30; }
QListWidget, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget {
    background-color: #2b2d31;
    border: 1px solid #3f4147;
    border-radius: 4px;
    padding: 4px;
    selection-background-color: #3d6fb4;
}
QListWidget::item { padding: 3px; }
QListWidget::item:selected { background-color: #3d6fb4; color: #ffffff; }
QProgressBar {
    background-color: #2b2d31;
    border: 1px solid #3f4147;
    border-radius: 4px;
    text-align: center;
    height: 18px;
}
QProgressBar::chunk { background-color: #3d8f5b; border-radius: 4px; }
QLabel#dropArea { border: 2px dashed #4a4c52; border-radius: 8px; color: #9a9c9f; padding: 14px; }
QSplitter::handle { background-color: #3f4147; }
QMenuBar { background-color: #2b2d31; color: #e6e6e6; }
QMenu { background-color: #2b2d31; color: #e6e6e6; border: 1px solid #3f4147; }
QMenu::item:selected { background-color: #3d6fb4; }
QTabWidget::pane { border: 1px solid #3f4147; }
QTabBar::tab { background: #2b2d31; padding: 6px 12px; border: 1px solid #3f4147; }
QTabBar::tab:selected { background: #3d6fb4; }
QCheckBox { spacing: 6px; }
QSlider::groove:horizontal { background: #3f4147; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal { background: #3d6fb4; width: 12px; margin: -5px 0; border-radius: 6px; }
QScrollBar:vertical { background: #1e1f22; width: 12px; }
QScrollBar::handle:vertical { background: #3f4147; border-radius: 5px; min-height: 24px; }
QHeaderView::section { background-color: #2b2d31; color: #e6e6e6; border: 1px solid #3f4147; padding: 4px; }
"""

LIGHT_STYLESHEET = """
QWidget {
    background-color: #f5f5f6;
    color: #202124;
    font-family: 'Segoe UI', sans-serif;
    font-size: 10.5pt;
}
QMainWindow { background-color: #f5f5f6; }
QPushButton {
    background-color: #ffffff;
    border: 1px solid #c7c8cc;
    border-radius: 4px;
    padding: 6px 14px;
}
QPushButton:hover { background-color: #ececee; }
QPushButton:pressed { background-color: #e0e1e4; }
QPushButton:disabled { color: #9a9a9d; background-color: #f0f0f1; border-color: #dcdcdf; }
QListWidget, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget {
    background-color: #ffffff;
    border: 1px solid #c7c8cc;
    border-radius: 4px;
    padding: 4px;
    selection-background-color: #3d6fb4;
    selection-color: #ffffff;
}
QListWidget::item { padding: 3px; }
QListWidget::item:selected { background-color: #3d6fb4; color: #ffffff; }
QProgressBar {
    background-color: #ffffff;
    border: 1px solid #c7c8cc;
    border-radius: 4px;
    text-align: center;
    height: 18px;
}
QProgressBar::chunk { background-color: #3d8f5b; border-radius: 4px; }
QLabel#dropArea { border: 2px dashed #b5b6ba; border-radius: 8px; color: #6a6b6e; padding: 14px; }
QSplitter::handle { background-color: #c7c8cc; }
QMenuBar { background-color: #ffffff; color: #202124; }
QMenu { background-color: #ffffff; color: #202124; border: 1px solid #c7c8cc; }
QMenu::item:selected { background-color: #3d6fb4; color: #ffffff; }
QTabWidget::pane { border: 1px solid #c7c8cc; }
QTabBar::tab { background: #ececee; padding: 6px 12px; border: 1px solid #c7c8cc; }
QTabBar::tab:selected { background: #3d6fb4; color: #ffffff; }
QCheckBox { spacing: 6px; }
QSlider::groove:horizontal { background: #c7c8cc; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal { background: #3d6fb4; width: 12px; margin: -5px 0; border-radius: 6px; }
QScrollBar:vertical { background: #f5f5f6; width: 12px; }
QScrollBar::handle:vertical { background: #c7c8cc; border-radius: 5px; min-height: 24px; }
QHeaderView::section { background-color: #ececee; color: #202124; border: 1px solid #c7c8cc; padding: 4px; }
"""


def stylesheet_for(dark: bool) -> str:
    return DARK_STYLESHEET if dark else LIGHT_STYLESHEET
