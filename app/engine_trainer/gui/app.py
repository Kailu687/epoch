"""Main window and application entry point (PySide6)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import psutil
from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtWidgets import (QApplication, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QStackedWidget,
                               QHBoxLayout, QWidget)

from .. import __version__
from ..io_csv import DatasetError
from ..workspace import Workspace, WorkspaceError
from . import screens as S

STYLE = """
QWidget{background:#0f1319;color:#e6edf3;font-size:13px}
QListWidget{background:#11161d;border:0;border-right:1px solid #252d38;padding:6px;outline:0}
QListWidget::item{padding:9px 10px;border-radius:6px;color:#9aa7b4}
QListWidget::item:selected{background:#1c2430;color:#4cc2ff}
QListWidget::item:disabled{color:#4b5563}
QPushButton{background:#1c2430;border:1px solid #2d3745;padding:7px 14px;border-radius:6px}
QPushButton:hover{border-color:#4cc2ff}QPushButton:disabled{color:#5b6673;background:#141a22}
QPushButton#primary{background:#4cc2ff;color:#04202e;border-color:#4cc2ff;font-weight:600}
QPushButton#primary:disabled{background:#27475a;color:#7b9aab}
QTableWidget,QTextEdit,QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox,QListWidget#flat{background:#151b24;border:1px solid #252d38;border-radius:6px}
QHeaderView::section{background:#1a212b;color:#9aa7b4;border:0;padding:5px}
QLabel#h1{font-size:20px;font-weight:600}QLabel#muted{color:#9aa7b4}
QLabel#warn{color:#d29922}QLabel#err{color:#f85149}QLabel#ok{color:#3fb950}
QLabel#planned{color:#a78bfa}
QStatusBar{background:#11161d;color:#9aa7b4}
QProgressBar{border:1px solid #252d38;border-radius:4px;background:#151b24;text-align:center}
QProgressBar::chunk{background:#4cc2ff}
"""

NAV = [("Welcome", "welcome"), ("Import dataset", "import"), ("Dataset explorer", "explorer"), ("Task", "task"),
       ("Models", "models"), ("Effort", "config"), ("Monitor", "monitor"), ("Results", "results"), ("Try model", "try"), ("Improve", "improve"),
       ("History", "history"), ("Registry", "registry"), ("Export", "export"), ("Settings", "settings")]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Epoch {__version__}")
        self.resize(1180, 760)
        self.ws: Workspace | None = None
        self.dataset_id = None
        self.experiment_id = None
        self.job = None
        self.settings = QSettings(os.environ.get("ET_SETTINGS_ORG", "Epoch"), "Epoch")
        self.cfg = {"target": None, "task_type": None, "seed": 42, "val_frac": 0.15, "test_frac": 0.15, "exclude": [],
                    "drop_duplicates": False, "models": None, "n_jobs": max(1, (psutil.cpu_count(logical=False) or 2) - 1),
                    "metric": None, "split_strategy": None, "group_column": None, "time_column": None, "budget_mode": "express",
                    "budget_seconds": None, "optimisation_profile": "balanced", "cv_folds": 5, "advice": None}
        self.nav = QListWidget()
        self.nav.setFixedWidth(190)
        self.stack = QStackedWidget()
        self.screens = {}
        for title, key in NAV:
            scr = getattr(S, key.capitalize() + "Screen")(self)
            self.screens[key] = scr
            self.stack.addWidget(scr)
            it = QListWidgetItem(title)
            it.setData(Qt.UserRole, key)
            self.nav.addItem(it)
        self.nav.currentRowChanged.connect(self._nav_changed)
        root = QWidget()
        lay = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.nav)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        self.status_ws = QLabel("No workspace")
        self.status_res = QLabel("")
        self.statusBar().addWidget(self.status_ws, 1)
        self.statusBar().addPermanentWidget(self.status_res)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(150)
        self._update_nav()
        self.go("welcome")

    # -------------------------------------------------------------- navigation / state
    def go(self, key):
        for i in range(self.nav.count()):
            if self.nav.item(i).data(Qt.UserRole) == key:
                self.nav.setCurrentRow(i)
                return

    def _nav_changed(self, row):
        if row < 0:
            return
        key = self.nav.item(row).data(Qt.UserRole)
        self.stack.setCurrentWidget(self.screens[key])
        if hasattr(self.screens[key], "refresh"):
            self.screens[key].refresh()

    def _update_nav(self):
        need_ws = {"import", "history", "registry", "settings", "improve"}
        need_ds = {"explorer", "task", "models", "config"}
        need_exp = {"monitor", "results", "export", "try"}
        for i in range(self.nav.count()):
            k = self.nav.item(i).data(Qt.UserRole)
            ok = k == "welcome" or (self.ws is not None and (k in need_ws or (k in need_ds and self.dataset_id) or
                                                              (k in need_exp and (self.experiment_id or k == "export"))))
            if k == "settings":
                ok = True
            flags = self.nav.item(i).flags()
            self.nav.item(i).setFlags(flags | Qt.ItemIsEnabled if ok else flags & ~Qt.ItemIsEnabled)

    def open_workspace(self, path):
        if self.job and not self.job.done:
            self.error("A training run is active. Cancel it before switching workspace.")
            return False
        try:
            if self.ws:
                self.ws.close()
            self.ws = Workspace(path)
        except (OSError, Exception) as e:
            self.error(f"Could not open workspace: {e}")
            return False
        recent = [p for p in self.settings.value("recent", [], list) if p != str(self.ws.root)]
        self.settings.setValue("recent", [str(self.ws.root)] + recent[:7])
        self.dataset_id = self.experiment_id = None
        self.status_ws.setText(f"Workspace: {self.ws.root}")
        self._update_nav()
        return True

    def set_dataset(self, did):
        self.dataset_id = did
        self.cfg.update(target=None, task_type=None, exclude=[], models=None, metric=None, split_strategy=None, group_column=None,
                        time_column=None, advice=None)
        self._update_nav()

    def error(self, msg):
        QMessageBox.warning(self, "Epoch", msg)

    def guard(self, fn, *a, **k):
        """Run fn, turning expected engine errors into dialogs."""
        try:
            return fn(*a, **k)
        except (DatasetError, WorkspaceError, FileExistsError, ValueError) as e:
            self.error(str(e))
        return None

    # -------------------------------------------------------------- training
    def start_training(self):
        if self.job and not self.job.done:
            return
        c = self.cfg
        self.experiment_id = self.ws.create_experiment(
            self.dataset_id, target=c["target"], task_type=c["task_type"], seed=c["seed"], val_frac=c["val_frac"], test_frac=c["test_frac"],
            exclude=c["exclude"], drop_duplicates=c["drop_duplicates"], n_jobs=c["n_jobs"], mode="automl", metric=c.get("metric"),
            split_strategy=c.get("split_strategy"), group_column=c.get("group_column"), time_column=c.get("time_column"),
            budget_mode=c.get("budget_mode"), budget_seconds=c.get("budget_seconds"), optimisation_profile=c.get("optimisation_profile"),
            cv_folds=c.get("cv_folds"), models=c.get("models"), advice=c.get("advice"))
        self.screens["monitor"].reset(self.experiment_id)
        self.job = self.ws.start(self.experiment_id)
        self._update_nav()
        self.go("monitor")

    def start_improvement(self, old_model_id, new_dataset_id):
        if self.job and not self.job.done:
            return
        eid = self.guard(self.ws.improve_model, old_model_id, new_dataset_id)
        if not eid:
            return
        self.improving_from = old_model_id
        self.experiment_id = eid
        self.screens["monitor"].reset(eid)
        self.job = self.ws.start(eid)
        self._update_nav()
        self.go("monitor")

    def cancel_training(self):
        if self.job and not self.job.done:
            for e in self.job.cancel():
                self._on_event(e)

    def _on_event(self, e):
        self.ws.handle_event(self.experiment_id, e)
        self.screens["monitor"].on_event(e)
        if e["type"] in ("finished", "failed", "cancelled"):
            self.screens["monitor"].on_done(e)

    def _tick(self):
        if self.job and not self.job.done:
            for e in self.job.poll(0):
                self._on_event(e)
            if self.job.done:
                self.job._proc.join(1)

    def closeEvent(self, ev):
        w = getattr(self.screens.get('import'), 'worker', None)
        if w and w.isRunning():
            w.wait(5000)
        if self.job and not self.job.done:
            self.cancel_training()
        if self.ws:
            self.ws.close()
        ev.accept()


def run() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    return app.exec()
