"""Shared widgets and helpers for the Epoch screens."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QFrame, QLabel, QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)


def lbl(text, name=None, wrap=True):
    l = QLabel(text)
    l.setWordWrap(wrap)
    if name:
        l.setObjectName(name)
    return l


def make_table(headers, stretch_last=True):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.verticalHeader().setVisible(False)
    t.horizontalHeader().setStretchLastSection(stretch_last)
    t.setMinimumHeight(150)
    return t


def fill(t: QTableWidget, rows):
    t.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            t.setItem(r, c, QTableWidgetItem("" if v is None else str(v)))
    t.resizeColumnsToContents()


def planned(text):
    l = QLabel(f"◌  {text} — planned, not available in this release")
    l.setObjectName("planned")
    return l


class BarChart(QWidget):
    """Horizontal bars of the primary validation metric per candidate. Values are real results; nothing is simulated."""

    def __init__(self):
        super().__init__()
        self.rows, self.title = [], ""
        self.setMinimumHeight(130)

    def set_data(self, rows, title, selected=None, baseline=None):
        self.rows, self.title, self.selected, self.baseline = rows, title, selected, baseline
        self.setMinimumHeight(40 + 28 * max(1, len(rows)))
        self.update()

    def paintEvent(self, _):
        from PySide6.QtGui import QPainter
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QColor("#9aa7b4"))
        p.drawText(0, 14, self.title)
        if not self.rows:
            return
        vmax = max(v for _, v in self.rows) or 1.0
        left, w = 150, max(50, self.width() - 230)
        for i, (name, v) in enumerate(self.rows):
            y = 26 + i * 28
            p.setPen(QColor("#e6edf3"))
            p.drawText(0, y + 14, name)
            p.setBrush(QColor("#21262d"))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(left, y + 2, w, 16, 3, 3)
            p.setBrush(QColor("#3fb950" if name == getattr(self, "selected", None) else "#4cc2ff"))
            p.drawRoundedRect(left, y + 2, max(2, int(w * v / vmax)), 16, 3, 3)
            p.setPen(QColor("#e6edf3"))
            p.drawText(left + w + 8, y + 14, f"{v:.4f}")
        if getattr(self, "baseline", None) is not None:
            x = left + int(w * self.baseline / vmax)
            p.setPen(QColor("#d29922"))
            p.drawLine(x, 22, x, 26 + 28 * len(self.rows))


class Page(QWidget):
    def __init__(self, win, title, subtitle=""):
        super().__init__()
        self.win = win
        # every page scrolls, so nothing is squashed on a small window; children keep a sensible minimum size
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self.root = QVBoxLayout(inner)
        self.root.setContentsMargins(24, 20, 24, 20)
        self.root.setSpacing(10)
        self.root.addWidget(lbl(title, "h1"))
        if subtitle:
            self.root.addWidget(lbl(subtitle, "muted"))

    def primary(self, text, slot):
        b = QPushButton(text)
        b.setObjectName("primary")
        b.clicked.connect(slot)
        return b

    def button(self, text, slot):
        b = QPushButton(text)
        b.clicked.connect(slot)
        return b
