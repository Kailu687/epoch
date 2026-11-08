"""All application screens. Controls for unimplemented features are shown disabled and marked (planned)."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import psutil
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QGuiApplication
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)

from .. import __version__
from ..evaluate import primary_metric
from ..io_csv import DatasetError
from ..workspace import Workspace
from .screens_v2 import ConfigScreen, ImproveScreen, ModelsScreen, MonitorScreen, ResultsScreen, TaskScreen, TryScreen  # noqa: F401


from .common import BarChart, Page, fill, lbl, make_table, planned  # noqa: F401


# ------------------------------------------------------------------ 1 welcome
class WelcomeScreen(Page):
    def __init__(self, win):
        super().__init__(win, "Welcome to Epoch", "Choose a workspace folder. Datasets, experiments and models are stored there on this computer.")
        row = QHBoxLayout()
        self.path = QLineEdit(str(Path.home() / "Documents" / "Epoch"))
        row.addWidget(self.path, 1)
        row.addWidget(self.button("Browse…", self.browse))
        self.root.addLayout(row)
        self.root.addWidget(self.primary("Open / create workspace", self.open))
        self.root.addWidget(lbl("Recent workspaces", "muted"))
        self.recent = QListWidget()
        self.recent.itemDoubleClicked.connect(lambda it: self.path.setText(it.text()) or self.open())
        self.root.addWidget(self.recent, 1)
        self.root.addWidget(lbl("Status: version %s — early development. Tabular CSV classification and regression are supported; "
                                "images, audio, text, time series and neural networks are planned." % __version__, "muted"))

    def refresh(self):
        self.recent.clear()
        self.recent.addItems(self.win.settings.value("recent", [], list))

    def browse(self):
        d = QFileDialog.getExistingDirectory(self, "Workspace folder", self.path.text())
        if d:
            self.path.setText(d)

    def open(self):
        if self.win.open_workspace(self.path.text().strip()):
            self.win.go("import")


# ------------------------------------------------------------------ 2 import
class ImportWorker(QThread):
    """Parses and profiles a CSV off the UI thread. Touches no SQLite connection; the main thread commits the result."""
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            self.succeeded.emit(Workspace.prepare_import(self.path))
        except DatasetError as e:
            self.failed.emit(str(e))
        except Exception as e:  # never let a worker exception vanish silently
            self.failed.emit(f"Unexpected error while reading the file: {type(e).__name__}: {e}")


class ImportScreen(Page):
    def __init__(self, win):
        super().__init__(win, "Import dataset", "Supported: CSV (comma, semicolon, tab or pipe separated). Excel, Parquet, images and audio are planned.")
        row = QHBoxLayout()
        self.path = QLineEdit()
        self.path.setPlaceholderText("Path to a .csv file")
        row.addWidget(self.path, 1)
        row.addWidget(self.button("Browse…", self.browse))
        self.root.addLayout(row)
        self.import_btn = self.primary("Import", lambda: self.do_import(self.path.text().strip()))
        self.root.addWidget(self.import_btn)
        self.root.addWidget(self.button("Create a synthetic demo dataset to try the workflow", self.make_sample))
        self.msg = lbl("")
        self.root.addWidget(self.msg)
        self.root.addWidget(lbl("Datasets in this workspace (double-click to use)", "muted"))
        self.table = make_table(["ID", "Name", "Rows", "Columns", "Imported"])
        self.table.cellDoubleClicked.connect(self.use_selected)
        self.root.addWidget(self.table, 1)
        for t in ("Excel / Parquet import", "Image folders", "Audio folders"):
            self.root.addWidget(planned(t))

    def refresh(self):
        if self.win.ws:
            self._rows = self.win.ws.datasets()
            fill(self.table, [(d["id"], d["name"], d["n_rows"], d["n_cols"], d["imported_at"]) for d in self._rows])

    def browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV files (*.csv *.tsv *.txt);;All files (*)")
        if p:
            self.path.setText(p)

    def do_import(self, path):
        """Start a background import. The UI stays responsive; the result arrives via _imported/_import_failed."""
        if not path:
            self.msg.setText("Choose a file first.")
            return
        if getattr(self, "worker", None) and self.worker.isRunning():
            return
        self.msg.setText("Reading and profiling the file…")
        self.import_btn.setEnabled(False)
        self.worker = ImportWorker(path)
        self.worker.succeeded.connect(self._imported)
        self.worker.failed.connect(self._import_failed)
        self.worker.finished.connect(lambda: self.import_btn.setEnabled(True))
        self.worker.start()

    def make_sample(self):
        """Write a generated (not real) customer-churn style CSV into the workspace and import it."""
        import numpy as np
        import pandas as pd
        rng = np.random.RandomState(7)
        n = 800
        df = pd.DataFrame({"customer_id": [f"C{i:05d}" for i in range(n)], "age": rng.randint(18, 80, n),
                           "monthly_spend": np.round(rng.gamma(3, 25, n), 2), "support_calls": rng.poisson(1.5, n),
                           "plan": rng.choice(["basic", "plus", "pro"], n, p=[0.5, 0.3, 0.2])})
        risk = 0.02 * df.monthly_spend + 0.5 * df.support_calls - 0.02 * df.age + (df.plan == "basic") * 0.8 + rng.randn(n) * 1.2
        df["churned"] = np.where(risk > np.percentile(risk, 70), "yes", "no")
        d = self.win.ws.root / "samples"
        d.mkdir(exist_ok=True)
        f = d / "synthetic_churn_demo.csv"
        df.to_csv(f, index=False)
        self.path.setText(str(f))
        self.do_import(str(f))

    def _imported(self, prep):
        did = self.win.guard(self.win.ws.commit_import, prep)
        if did:
            self.msg.setText("Imported.")
            self.win.set_dataset(did)
            self.win.go("explorer")
        else:
            self.msg.setText("Import failed (see message).")

    def _import_failed(self, message):
        self.msg.setText("Import failed.")
        self.win.error(message)

    def use_selected(self, row, _):
        self.win.set_dataset(self._rows[row]["id"])
        self.win.go("explorer")


# ------------------------------------------------------------------ 3 explorer
class ExplorerScreen(Page):
    def __init__(self, win):
        super().__init__(win, "Dataset explorer")
        self.summary = lbl("")
        self.root.addWidget(self.summary)
        self.root.addWidget(lbl("Preview (first 50 rows)", "muted"))
        self.preview = QTableWidget()
        self.preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.root.addWidget(self.preview, 3)
        self.root.addWidget(lbl("Column profile", "muted"))
        self.cols = make_table(["Column", "Kind", "Type", "Missing %", "Unique", "Notes"])
        self.root.addWidget(self.cols, 3)
        self.root.addWidget(self.primary("Continue to task", lambda: win.go("task")))

    def refresh(self):
        ws, did = self.win.ws, self.win.dataset_id
        if not did:
            return
        ds = ws.dataset(did)
        p = ds["profile"]
        self.summary.setText(f"{ds['name']} — {p['n_rows']:,} rows × {p['n_cols']} columns, {p['n_duplicate_rows']} duplicate rows. SHA-256 {ds['sha256'][:16]}…")
        df = ws.preview(did)
        self.preview.setColumnCount(len(df.columns))
        self.preview.setHorizontalHeaderLabels(list(df.columns))
        self.preview.setRowCount(len(df))
        for r in range(len(df)):
            for c in range(len(df.columns)):
                v = df.iat[r, c]
                self.preview.setItem(r, c, QTableWidgetItem("" if v != v else (f"{v:.5g}" if isinstance(v, float) else str(v))))
        self.preview.resizeColumnsToContents()
        fill(self.cols, [(c["name"], c["kind"], c["dtype"], c["pct_missing"], c["n_unique"], "; ".join(c["notes"])) for c in p["columns"]])


# ------------------------------------------------------------------ 9 history
class HistoryScreen(Page):
    def __init__(self, win):
        super().__init__(win, "Experiment history", "Double-click an experiment to open its results.")
        self.table = make_table(["ID", "Dataset", "Status", "Task", "Created"])
        self.table.cellDoubleClicked.connect(self.open)
        self.root.addWidget(self.table, 1)
        self._rows = []

    def refresh(self):
        if not self.win.ws:
            return
        self._rows = self.win.ws.experiments()
        fill(self.table, [(r["id"], r["dataset"], r["status"], json.loads(r["task_json"])["task_type"], r["created_at"]) for r in self._rows])

    def open(self, row, _):
        self.win.experiment_id = self._rows[row]["id"]
        self.win._update_nav()
        self.win.go("results")


# ------------------------------------------------------------------ 10 registry
class RegistryScreen(Page):
    def __init__(self, win):
        super().__init__(win, "Model registry", "Models produced by experiments in this workspace.")
        self.table = make_table(["Model ID", "Experiment", "Algorithm", "Selected", "Test uses", "Created"])
        self.root.addWidget(self.table, 1)
        self.root.addWidget(self.button("Export selected row…", self.export_sel))
        self._rows = []

    def refresh(self):
        if not self.win.ws:
            return
        self._rows = self.win.ws.models()
        fill(self.table, [(m["id"], m["experiment_id"], m["estimator"], "yes" if m["selected"] else "", m["test_eval_count"], m["created_at"]) for m in self._rows])

    def export_sel(self):
        r = self.table.currentRow()
        if r >= 0:
            self.win.screens["export"].preselect = self._rows[r]["id"]
            self.win.go("export")


# ------------------------------------------------------------------ 11 export
class ExportScreen(Page):
    def __init__(self, win):
        super().__init__(win, "Export model", "Writes the pipeline, schema, labels, config, metrics, requirements and an inference example.")
        self.model = QComboBox()
        self.root.addWidget(self.model)
        row = QHBoxLayout()
        self.dest = QLineEdit()
        self.model.currentIndexChanged.connect(self._default_dest)
        row.addWidget(self.dest, 1)
        row.addWidget(self.button("Browse…", self.browse))
        self.root.addLayout(row)
        self.root.addWidget(lbl("Security: the exported model file is pickle-based. Loading it can execute code. Only load bundles from sources you trust; "
                                "manifest.json detects modification, not malicious origin.", "warn"))
        self.root.addWidget(self.primary("Export bundle", lambda: self.do_export(self.dest.text().strip())))
        self.out = lbl("")
        self.root.addWidget(self.out)
        for t in ("ONNX export", "Safe (non-pickle) model format"):
            self.root.addWidget(planned(t))
        self.root.addStretch(1)
        self.preselect = None

    def refresh(self):
        self.model.clear()
        ms = self.win.ws.models()
        ms.sort(key=lambda m: (m["experiment_id"] != self.win.experiment_id, not m["selected"]))
        for m in ms:
            self.model.addItem(f"{m['id']} — {m['estimator']}{' (selected)' if m['selected'] else ''}", m["id"])
        if self.preselect:
            i = self.model.findData(self.preselect)
            if i >= 0:
                self.model.setCurrentIndex(i)
            self.preselect = None

    def _default_dest(self, *_):
        mid = self.model.currentData()
        if mid and self.win.ws:
            self.dest.setText(str(self.win.ws.root / "exports" / mid))

    def browse(self):
        d = QFileDialog.getExistingDirectory(self, "Export folder")
        if d:
            self.dest.setText(str(Path(d) / (self.model.currentData() or "export")))

    def do_export(self, dest):
        mid = self.model.currentData()
        if not mid or not dest:
            self.out.setText("Choose a model and a destination folder.")
            return
        mf = self.win.guard(self.win.ws.export, mid, dest)
        self.out.setText(f"Exported {len(mf['files']) + 1} files (including manifest.json) to {dest}" if mf else "Export failed.")


# ------------------------------------------------------------------ 12 settings
class SettingsScreen(Page):
    def __init__(self, win):
        super().__init__(win, "Settings and diagnostics")
        self.info = QTextEdit(); self.info.setReadOnly(True)
        self.root.addWidget(self.info, 1)
        self.root.addWidget(self.button("Copy diagnostics to clipboard", self.copy))
        self.root.addWidget(planned("Telemetry (none is collected; no opt-in exists)"))
        self.root.addWidget(planned("GPU selection"))

    def text(self):
        from ..export import env_versions
        v = psutil.virtual_memory()
        lines = [f"Epoch {__version__}", f"Workspace: {self.win.ws.root if self.win.ws else '(none)'}",
                 f"CPU: {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count()} logical cores",
                 f"RAM: {v.total / 2**30:.1f} GiB total, {v.available / 2**30:.1f} GiB available",
                 "GPU: discovery not implemented in this release (planned)"]
        lines += [f"{k}: {x}" for k, x in env_versions().items()]
        return "\n".join(lines)

    def refresh(self):
        self.info.setPlainText(self.text())

    def copy(self):
        QGuiApplication.clipboard().setText(self.text())
