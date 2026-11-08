"""Epoch v2 screens: Let Epoch decide (task), model families, effort, live search, results with advice, and Try model."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import psutil
from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QBrush, QColor, QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QScrollArea,
                               QSpinBox, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)

from ..automl import metrics as mx
from ..automl import zoo
from .common import BarChart, Page, fill, lbl, make_table, planned

CLS_METRICS = ["macro_f1", "balanced_accuracy", "accuracy", "roc_auc", "pr_auc"]
REG_METRICS = ["rmse", "mae", "r2"]
EFFORT = [("Express (about 2-3 minutes)", "express"), ("Balanced (about 20-30 minutes)", "balanced"), ("Deep (up to 2 hours)", "deep")]
STYLE = [("Balanced", "balanced"), ("Best accuracy", "accuracy_first"), ("Fastest", "speed_first"),
         ("Small and light (edge devices)", "lightweight"), ("Easy to explain", "interpretable")]


def _data_col(combo: QComboBox, value):
    i = combo.findData(value)
    combo.setCurrentIndex(i if i >= 0 else 0)


# ------------------------------------------------------------------ task: Let Epoch decide
class TaskScreen(Page):
    def __init__(self, win):
        super().__init__(win, "Task", "Describe your data in plain words and Epoch will choose the setup, or choose everything yourself. "
                                      "Not sure? Leave it empty, or write \"I don't know\".")
        self.desc = QTextEdit()
        self.desc.setPlaceholderText("Example: Customer records. I want to know who is likely to leave.\n"
                                     "Or: Daily sales from a shop; forecast next month.  Or leave this empty.")
        self.desc.setFixedHeight(78)
        self.root.addWidget(self.desc)
        row = QHBoxLayout()
        self.ask_btn = self.primary("Let Epoch decide", self.ask)
        row.addWidget(self.ask_btn)
        row.addStretch(1)
        self.root.addLayout(row)
        self.advice_box = QTextEdit()
        self.advice_box.setReadOnly(True)
        self.advice_box.setMinimumHeight(190)
        self.advice_box.setPlaceholderText("Epoch's reasoning will appear here.")
        self.root.addWidget(self.advice_box)
        self.q_holder = QVBoxLayout()
        self.root.addLayout(self.q_holder)
        self.apply_btn = self.button("Apply my answers", self.answer)
        self.apply_btn.setVisible(False)
        self.root.addWidget(self.apply_btn)

        self.root.addWidget(lbl("Your choices (filled in by Epoch, change anything you like)", "muted"))
        form = QFormLayout()
        self.target = QComboBox(); self.target.currentIndexChanged.connect(self.on_target)
        self.task = QComboBox(); self.task.addItem("classification (pick a category)", "classification"); self.task.addItem("regression (predict a number)", "regression")
        self.task.currentIndexChanged.connect(self.sync)
        self.group = QComboBox(); self.group.currentIndexChanged.connect(self.sync)
        self.time = QComboBox(); self.time.currentIndexChanged.connect(self.sync)
        form.addRow("What to predict (target)", self.target)
        form.addRow("Task type", self.task)
        form.addRow("Same person/patient/machine in many rows? (group column)", self.group)
        form.addRow("Rows are ordered in time? (time column)", self.time)
        self.root.addLayout(form)
        self.why = lbl("", "muted")
        self.root.addWidget(self.why)
        self.excl = QListWidget(); self.excl.setObjectName("flat"); self.excl.setMaximumHeight(110)
        self.excl.itemChanged.connect(lambda *_: self.sync())
        self.root.addWidget(lbl("Columns to leave out", "muted"))
        self.root.addWidget(self.excl)
        self.dedupe = QCheckBox("Remove duplicate rows before splitting")
        self.dedupe.toggled.connect(lambda *_: self.sync())
        self.root.addWidget(self.dedupe)
        self.root.addWidget(self.button("Validate", self.validate))
        self.findings = QListWidget(); self.findings.setObjectName("flat"); self.findings.setMaximumHeight(120)
        self.root.addWidget(self.findings)
        self.next = self.primary("Continue", lambda: win.go("models"))
        self.next.setEnabled(False)
        self.root.addWidget(self.next)
        self._loading = False
        self._advice = None
        self._qcombos = {}

    # -- data
    def refresh(self):
        ds = self.win.ws.dataset(self.win.dataset_id)
        cols = ds["profile"]["columns"]
        sugg = [s["column"] for s in ds["profile"]["target_suggestions"]]
        self._loading = True
        self.target.clear(); self.group.clear(); self.time.clear(); self.excl.clear()
        for c in sugg + [c["name"] for c in cols if c["name"] not in sugg]:
            self.target.addItem(c, c)
        self.group.addItem("(none)", None); self.time.addItem("(none)", None)
        for c in cols:
            self.group.addItem(c["name"], c["name"]); self.time.addItem(c["name"], c["name"])
            it = QListWidgetItem(f"{c['name']}  [{c['kind']}]")
            it.setData(Qt.UserRole, c["name"])
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.excl.addItem(it)
        self._loading = False
        self.findings.clear(); self.advice_box.clear(); self._advice = None
        self._clear_questions()
        self.next.setEnabled(False)
        self.on_target()

    def on_target(self, *_):
        if self._loading or self.target.currentData() is None:
            return
        s = self.win.guard(self.win.ws.suggest_task, self.win.dataset_id, self.target.currentData())
        if s:
            self._loading = True
            _data_col(self.task, s.task_type)
            self._loading = False
            ask = " Please confirm this choice." if s.needs_confirmation else ""
            self.why.setText(f"Data suggests: {s.task_type} ({s.confidence} confidence) — {'; '.join(s.reasons)}.{ask}")
        self.sync()

    def sync(self, *_):
        if self._loading:
            return
        ex = [self.excl.item(i).data(Qt.UserRole) for i in range(self.excl.count()) if self.excl.item(i).checkState() == Qt.Checked]
        g, t = self.group.currentData(), self.time.currentData()
        c = self.win.cfg
        c.update(target=self.target.currentData(), task_type=self.task.currentData(), exclude=ex, drop_duplicates=self.dedupe.isChecked(),
                 group_column=g, time_column=t, models=None)
        c["split_strategy"] = "temporal" if t else "group" if g else ("stratified" if c["task_type"] == "classification" else "random")
        if self._advice is None or self._advice.get("target") != c["target"] or self._advice.get("task_type") != c["task_type"]:
            c["advice"] = None  # manual override: do not claim Epoch chose this
            if self._advice is not None:
                c["metric"] = None
        self.next.setEnabled(False)

    # -- Let Epoch decide
    def ask(self):
        w = self.win
        w.setCursor(Qt.WaitCursor)
        try:
            adv = w.guard(w.ws.advise, w.dataset_id, self.desc.toPlainText().strip())
        finally:
            w.unsetCursor()
        if adv is not None:
            self._show_advice(adv)

    def _clear_questions(self):
        while self.q_holder.count():
            it = self.q_holder.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
            elif it.layout():
                while it.layout().count():
                    x = it.layout().takeAt(0)
                    if x.widget():
                        x.widget().deleteLater()
        self._qcombos = {}
        self.apply_btn.setVisible(False)

    def _show_advice(self, adv):
        from ..advisor import describe_advice
        self._advice_obj = adv
        self._advice = adv.to_dict()
        self.advice_box.setPlainText(describe_advice(adv))
        self._clear_questions()
        for q in adv.questions:
            row = QHBoxLayout()
            row.addWidget(lbl(q["text"]), 3)
            cb = QComboBox()
            for o in q.get("options", []):
                cb.addItem(str(o), str(o))
            self._qcombos[q["id"]] = cb
            row.addWidget(cb, 2)
            wrapper = QWidget(); wrapper.setLayout(row)
            self.q_holder.addWidget(wrapper)
        self.apply_btn.setVisible(bool(adv.questions))
        self._apply_to_widgets(adv)

    def answer(self):
        from ..advisor import apply_answers
        answers = {qid: cb.currentData() for qid, cb in self._qcombos.items()}
        try:
            adv = apply_answers(self._advice_obj, answers)
        except ValueError as e:
            self.win.error(str(e))
            return
        self._show_advice(adv)

    def _apply_to_widgets(self, adv):
        """Fill the manual controls from the advice and store everything the engine needs in the shared config."""
        self._loading = True
        if adv.target:
            _data_col(self.target, adv.target)
        if adv.task_type in ("classification", "regression"):
            _data_col(self.task, adv.task_type)
        _data_col(self.group, adv.group_column)
        _data_col(self.time, adv.time_column)
        for i in range(self.excl.count()):
            it = self.excl.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in (adv.exclude_columns or []) else Qt.Unchecked)
        self._loading = False
        c = self.win.cfg
        c.update(target=adv.target, task_type=adv.task_type if adv.task_type != "unknown" else self.task.currentData(),
                 exclude=list(adv.exclude_columns or []), metric=adv.metric, split_strategy=adv.split_strategy, group_column=adv.group_column,
                 time_column=adv.time_column, budget_mode=adv.budget_mode, optimisation_profile=adv.optimisation_profile, advice=self._advice, models=None)
        self.next.setEnabled(False)
        if adv.target and adv.task_type in ("classification", "regression") and not adv.questions:
            self.validate()

    def validate(self):
        c = self.win.cfg
        if not c.get("target"):
            self.win.error("Choose what to predict first (or let Epoch decide).")
            return
        ex = list(c["exclude"]) + [x for x in (c.get("group_column"), c.get("time_column")) if x]
        vr = self.win.guard(self.win.ws.validate_preview, self.win.dataset_id, c["target"], c["task_type"], ex, c["drop_duplicates"])
        if vr is None:
            return
        self.findings.clear()
        for f in vr.findings:
            it = QListWidgetItem(f"[{f.severity.upper()}] {f.message}")
            it.setForeground(QBrush(QColor({"error": "#f85149", "warning": "#d29922"}.get(f.severity, "#9aa7b4"))))
            self.findings.addItem(it)
        self.findings.addItem(QListWidgetItem(f"{len(vr.features)} column(s) will be used as inputs. A leakage check also runs during training."))
        self.next.setEnabled(vr.ok)


# ------------------------------------------------------------------ models
class ModelsScreen(Page):
    def __init__(self, win):
        super().__init__(win, "Models", "Epoch tries these families automatically and keeps the ones that work best on your data. "
                                         "Untick any you want to exclude.")
        self.list = QListWidget(); self.list.setObjectName("flat")
        self.list.itemChanged.connect(lambda *_: self.sync())
        self.root.addWidget(self.list, 1)
        self.note = lbl("", "muted")
        self.root.addWidget(self.note)
        for t in ("Neural networks (PyTorch)", "Fine-tune a pretrained model", "Custom model plugin"):
            self.root.addWidget(planned(t))
        self.root.addWidget(self.primary("Continue", lambda: win.go("config")))
        self._loading = False

    def refresh(self):
        c = self.win.cfg
        n = self.win.ws.dataset(self.win.dataset_id)["n_rows"]
        defs = zoo.available(c["task_type"], int(n * 0.85), c.get("optimisation_profile") or "balanced")
        self._loading = True
        self.list.clear()
        for d in defs:
            it = QListWidgetItem(f"{d.label} — {d.description}")
            it.setData(Qt.UserRole, d.id)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            if d.id == "baseline":
                it.setFlags(it.flags() & ~Qt.ItemIsEnabled)  # the do-nothing baseline always runs
            self.list.addItem(it)
        self._loading = False
        missing = [m for m, ok in (("LightGBM", zoo.HAS_LIGHTGBM), ("Optuna (smart tuning)", zoo.HAS_OPTUNA)) if not ok]
        self.note.setText("Not installed, so skipped: " + ", ".join(missing) if missing else
                          "Large datasets automatically skip slow families (support vector machines, nearest neighbours).")
        self.sync()

    def sync(self):
        if self._loading:
            return
        ids = [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count()) if self.list.item(i).checkState() == Qt.Checked]
        total = self.list.count()
        self.win.cfg["models"] = None if len(ids) == total else ids


# ------------------------------------------------------------------ config
class ConfigScreen(Page):
    def __init__(self, win):
        super().__init__(win, "Effort and settings", "How long Epoch may search, and what matters most. The defaults suit most data.")
        form = QFormLayout()
        self.effort = QComboBox()
        for t, v in EFFORT:
            self.effort.addItem(t, v)
        self.style_combo = QComboBox()
        for t, v in STYLE:
            self.style_combo.addItem(t, v)
        self.metric_combo = QComboBox()
        self.seconds = QSpinBox(); self.seconds.setRange(0, 86400); self.seconds.setSpecialValueText("automatic"); self.seconds.setSuffix(" s")
        self.seed = QSpinBox(); self.seed.setRange(0, 2**31 - 1); self.seed.setValue(42)
        self.val = QDoubleSpinBox(); self.val.setRange(0.05, 0.4); self.val.setSingleStep(0.05); self.val.setValue(0.15)
        self.test = QDoubleSpinBox(); self.test.setRange(0.05, 0.4); self.test.setSingleStep(0.05); self.test.setValue(0.15)
        self.jobs = QSpinBox(); self.jobs.setRange(1, max(1, psutil.cpu_count() or 1)); self.jobs.setValue(win.cfg["n_jobs"])
        self.folds = QSpinBox(); self.folds.setRange(3, 10); self.folds.setValue(5)
        for sp in (self.seconds, self.seed, self.val, self.test, self.jobs, self.folds):
            sp.setMaximumWidth(180)
        form.addRow("Effort", self.effort)
        form.addRow("Time limit override", self.seconds)
        form.addRow("What matters most", self.style_combo)
        form.addRow("Score to optimise", self.metric_combo)
        form.addRow("Random seed", self.seed)
        form.addRow("Cross-validation folds", self.folds)
        form.addRow("Validation fraction", self.val)
        form.addRow("Test fraction (locked, used once)", self.test)
        form.addRow("CPU workers", self.jobs)
        self.root.addLayout(form)
        self.plan_text = lbl("", "muted")
        self.root.addWidget(self.plan_text)
        self.root.addWidget(lbl("Models are compared by cross-validation on the development data. The test set is never used to choose or tune anything.", "muted"))
        for t in ("Device: GPU", "Pause and resume", "Memory and latency limits"):
            self.root.addWidget(planned(t))
        self.root.addStretch(1)
        self.start = self.primary("Start training", self.go_train)
        self.root.addWidget(self.start)

    def refresh(self):
        c = self.win.cfg
        _data_col(self.effort, c.get("budget_mode") or "express")
        _data_col(self.style_combo, c.get("optimisation_profile") or "balanced")
        self.metric_combo.clear()
        self.metric_combo.addItem("Let Epoch choose", None)
        for m in (CLS_METRICS if c.get("task_type") == "classification" else REG_METRICS):
            self.metric_combo.addItem(m, m)
        _data_col(self.metric_combo, c.get("metric"))
        strat = c.get("split_strategy") or "stratified"
        self.plan_text.setText(f"Target: {c.get('target')} ({c.get('task_type')}). Split: {strat}"
                               + (f" by {c['group_column']}" if c.get("group_column") and strat == "group" else "")
                               + (f" over {c['time_column']}" if c.get("time_column") and strat == "temporal" else "") + ".")

    def go_train(self):
        c = self.win.cfg
        if not c["target"]:
            self.win.error("Choose a target on the Task screen first.")
            return
        if self.val.value() + self.test.value() >= 0.7:
            self.win.error("Validation + test fractions must be below 70%.")
            return
        c.update(seed=self.seed.value(), val_frac=self.val.value(), test_frac=self.test.value(), n_jobs=self.jobs.value(), cv_folds=self.folds.value(),
                 budget_mode=self.effort.currentData(), optimisation_profile=self.style_combo.currentData(), metric=self.metric_combo.currentData(),
                 budget_seconds=self.seconds.value() or None)
        ex = list(c["exclude"]) + [x for x in (c.get("group_column"), c.get("time_column")) if x]
        vr = self.win.guard(self.win.ws.validate_preview, self.win.dataset_id, c["target"], c["task_type"], ex, c["drop_duplicates"])
        if vr is None:
            return
        if not vr.ok:
            self.win.error("Validation errors must be fixed first:\n- " + "\n- ".join(f.message for f in vr.errors))
            return
        self.win.start_training()


# ------------------------------------------------------------------ monitor
class MonitorScreen(Page):
    def __init__(self, win):
        super().__init__(win, "Active experiment")
        self.stage = lbl("No run started.")
        self.root.addWidget(self.stage)
        self.table = make_table(["Model", "Status", "Time (s)", "Cross-validation score"])
        self.root.addWidget(self.table, 2)
        self.bar = QProgressBar(); self.bar.setRange(0, 1)
        self.root.addWidget(self.bar)
        self.res = lbl("Resources: —", "muted")
        self.root.addWidget(self.res)
        self.root.addWidget(lbl("GPU / VRAM monitoring: not available in this release (planned).", "planned"))
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(160)
        self.root.addWidget(self.log, 2)
        row = QHBoxLayout()
        self.cancel = self.button("Cancel run", win.cancel_training)
        self.results = self.primary("View results", lambda: win.go("results"))
        self.results.setEnabled(False)
        row.addWidget(self.cancel); row.addStretch(1); row.addWidget(self.results)
        self.root.addLayout(row)
        self._rows = {}

    def reset(self, eid):
        self.stage.setText(f"Run {eid}: starting worker process…")
        self.table.setRowCount(0)
        self._rows = {}
        self.bar.setRange(0, 0)  # busy: search time is open-ended
        self.log.clear()
        self.cancel.setEnabled(True)
        self.results.setEnabled(False)

    def _set(self, cid, status=None, secs=None, metric=None):
        if cid not in self._rows:
            self._rows[cid] = self.table.rowCount()
            self.table.insertRow(self.table.rowCount())
            self.table.setItem(self._rows[cid], 0, QTableWidgetItem(cid))
        r = self._rows[cid]
        for col, v in ((1, status), (2, secs), (3, metric)):
            if v is not None:
                self.table.setItem(r, col, QTableWidgetItem(str(v)))

    @staticmethod
    def _nat(metric, v):
        return -v if metric in mx.LOWER_IS_BETTER else v

    def on_event(self, e):
        t = e["type"]
        if t == "resource":
            self.res.setText(f"CPU {e['cpu_percent']:.0f}%   RAM {e['ram_percent']:.0f}%   worker memory {e['worker_rss_mb']:.0f} MB")
            self.win.status_res.setText(f"CPU {e['cpu_percent']:.0f}% | RAM {e['ram_percent']:.0f}%")
            return
        self._metric = getattr(self, "_metric", "score")
        if t == "stage":
            self.stage.setText(f"Stage: {e['name']}")
        elif t == "findings":
            for f in e["findings"]:
                self.log.append(f"[{f['severity']}] {f['message']}")
        elif t == "split":
            self.log.append(f"Split {e['sizes']} ({e.get('strategy', 'stratified')}, hash {e['hash'][:12]})")
        elif t == "plan":
            self._metric = e.get("metric", "score")
            self.log.append(f"Score: {self._metric}; cross-validation: {e.get('scheme')}; time budget {e.get('budget_seconds'):.0f} s")
            for c in e["candidates"]:
                self._set(c["id"], "queued")
        elif t == "candidate_start":
            self._set(e["id"], e.get("stage", "running") + "…")
            self.stage.setText(f"{e.get('stage', 'Training').capitalize()}: {e['id']}")
        elif t == "screened":
            self._set(e["id"], "screened", e["seconds"], f"{self._nat(self._metric, e['cv_mean']):.4f} ± {e['cv_std']:.4f}")
        elif t == "candidate_skipped":
            self._set(e["id"], "skipped (time budget)")
        elif t == "trial":
            self.stage.setText(f"Tuning {e['id']}: trial {e['number'] + 1}")
        elif t == "validated":
            self._set(e["id"], "tuned", None, f"{self._nat(self._metric, e['cv_mean']):.4f} ± {e['cv_std']:.4f}")
        elif t == "candidate_add":
            self._set(e["id"], "queued")
        elif t == "candidate_done":
            v = e["val"]
            cv = f"{self._nat(self._metric, v['cv_mean']):.4f} ± {v.get('cv_std', 0):.4f}" if "cv_mean" in v else None
            self._set(e["id"], "ready", e["fit_seconds"], cv)
        elif t == "ensemble":
            self.log.append(f"Ensemble of {', '.join(m for m, _ in e['members'])}: gain {e['gain']:+.4f} (needs > {e['margin']:.4f}) — {'kept' if e['accepted'] else 'not kept'}")
        elif t == "candidate_failed":
            self._set(e["id"], "FAILED", None, e["error"])
            self.log.append(f"{e['id']} failed: {e['error']}")
        elif t == "recommendations":
            for r in e["items"][:5]:
                self.log.append(f"[advice:{r['severity']}] {r['title']}")
        elif t == "selected":
            self.log.append(f"Chosen: {e['id']}. {e.get('reason', '')}")
        elif t == "failed":
            self.log.append(f"FAILED: {e['reason']}")

    def on_done(self, e):
        self.cancel.setEnabled(False)
        self.bar.setRange(0, 1); self.bar.setValue(1 if e["type"] == "finished" else 0)
        self.stage.setText({"finished": "Finished.", "failed": "Run failed (see log).", "cancelled": "Cancelled."}[e["type"]])
        self.results.setEnabled(True)
        self.win.screens["history"].refresh()


# ------------------------------------------------------------------ results
class ResultsScreen(Page):
    def __init__(self, win):
        super().__init__(win, "Results and advice")
        self.summary = lbl("")
        self.root.addWidget(self.summary)
        self.why = lbl("", "muted")
        self.root.addWidget(self.why)
        self.chart = BarChart()
        self.root.addWidget(self.chart)
        self.table = make_table(["Model", "Status", "CV score ± std", "Tuned", "Time (s)", ""])
        self.table.setMinimumHeight(120)
        self.root.addWidget(self.table, 2)
        self.root.addWidget(lbl("How to improve this model (from the measured results)", "muted"))
        self.recs = QTextEdit(); self.recs.setReadOnly(True); self.recs.setMinimumHeight(160)
        self.root.addWidget(self.recs, 2)
        self.root.addWidget(lbl("Where it is right and wrong (cross-validated predictions; rows = true, columns = predicted)", "muted"))
        self.cm = make_table([])
        self.root.addWidget(self.cm, 2)
        self.test_btn = self.primary("Evaluate chosen model on the locked test set (single use)", self.eval_test)
        self.root.addWidget(self.test_btn)
        self.test_out = QTextEdit(); self.test_out.setReadOnly(True); self.test_out.setMinimumHeight(110)
        self.root.addWidget(self.test_out, 1)
        row = QHBoxLayout()
        row.addWidget(self.button("Open full report", self.open_report))
        row.addWidget(self.button("Try this model", lambda: win.go("try")))
        row.addWidget(self.button("Export", lambda: win.go("export")))
        self.root.addLayout(row)
        self.mid = None

    def refresh(self):
        w = self.win
        if not w.experiment_id:
            return
        exp = w.ws.experiment(w.experiment_id)
        summ = w.ws.summary(w.experiment_id)
        task = exp["task"]["task_type"]
        self.mid = w.ws.selected_model(w.experiment_id)
        if exp["status"] != "finished" or not summ:
            self.summary.setText(f"Run status: {exp['status']}. {exp['error'] or ''}")
            self.why.setText("")
            self.test_btn.setEnabled(False)
            return
        metric, lower = summ["selection_metric"], summ["selection_metric"] in mx.LOWER_IS_BETTER
        nat = lambda v: None if v is None else (-v if lower else v)
        sel_key = self.mid.split("-", 2)[-1] if self.mid else None
        cands = {c["cand_key"]: c for c in exp["candidates"]}
        sel = (cands.get(sel_key) or {}).get("val") or {}
        base = (cands.get("baseline") or {}).get("val") or {}
        msg = f"Chosen: {sel_key} — cross-validated {metric} {nat(sel.get('cv_mean')):.4f} ± {sel.get('cv_std', 0):.4f}."
        if base.get("cv_mean") is not None:
            msg += f" Baseline {nat(base['cv_mean']):.4f}. " + ("It clearly beats the baseline." if summ.get("beats_baseline") else "It does NOT clearly beat the baseline.")
        self.summary.setText(msg)
        self.why.setText(summ.get("selection_reason", ""))
        rows, bars = [], []
        for k, c in cands.items():
            v = c["val"] or {}
            cvs = f"{nat(v['cv_mean']):.4f} ± {v.get('cv_std', 0):.4f}" if v.get("cv_mean") is not None else ""
            rows.append((k, c["status"] + (f": {c['error']}" if c["error"] else ""), cvs, "yes" if v.get("tuned") else "", c["fit_seconds"] or "",
                         "◀ chosen" if k == sel_key else ""))
            if v.get("cv_mean") is not None:
                bars.append((k, abs(nat(v["cv_mean"]))))
        fill(self.table, rows)
        self.chart.set_data(bars, f"Cross-validated {metric} ({'lower' if lower else 'higher'} is better); yellow line = baseline", sel_key,
                            None if base.get("cv_mean") is None else abs(nat(base["cv_mean"])))
        recs = summ.get("recommendations") or []
        self.recs.setPlainText("\n\n".join(f"[{r['severity'].upper()}] {r['title']}\n{r['detail']}\n  - " + "\n  - ".join(r["actions"]) for r in recs)
                               or "No issues were detected by the automatic checks. That is not a guarantee of quality.")
        self.cm.clear(); self.cm.setRowCount(0); self.cm.setColumnCount(0)
        if task == "classification" and sel.get("confusion_matrix"):
            m = sel["confusion_matrix"]; n = len(m["labels"])
            self.cm.setColumnCount(n); self.cm.setRowCount(n)
            self.cm.setHorizontalHeaderLabels([str(x) for x in m["labels"]]); self.cm.setVerticalHeaderLabels([str(x) for x in m["labels"]])
            self.cm.verticalHeader().setVisible(True)
            for i in range(n):
                for j in range(n):
                    self.cm.setItem(i, j, QTableWidgetItem(str(m["matrix"][i][j])))
        self.test_btn.setEnabled(bool(self.mid))
        mrow = next((x for x in exp["models"] if x["id"] == self.mid), None)
        self.test_out.setPlainText(self._fmt_test(json.loads(mrow["test_metrics_json"]), mrow["test_eval_count"]) if mrow and mrow["test_metrics_json"]
                                   else "The locked test set has not been used yet.")

    @staticmethod
    def _fmt_test(rep, count):
        keys = [k for k, v in rep.items() if isinstance(v, (int, float)) and k != "n"]
        return f"Locked test set (n={rep['n']}, evaluated {count} time(s)):\n" + "\n".join(f"  {k}: {rep[k]:.4f}" for k in keys)

    def eval_test(self):
        w = self.win
        ack = False
        m = w.ws.store.one("SELECT test_eval_count c FROM models WHERE id=?", self.mid)
        if m["c"]:
            if QMessageBox.question(self, "Test set already used", "The test set has already been used for this model. Using it again biases the estimate and will be recorded. Continue?") != QMessageBox.Yes:
                return
            ack = True
        w.guard(w.ws.evaluate_test, self.mid, acknowledge_repeat=ack)
        self.refresh()

    def open_report(self):
        if not self.mid:
            return
        html = self.win.guard(self.win.ws.report_html, self.mid)
        if html:
            p = Path(tempfile.gettempdir()) / f"epoch_report_{self.mid}.html"
            p.write_text(html, encoding="utf-8")
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))


# ------------------------------------------------------------------ try model
class RowDialog(QDialog):
    def __init__(self, parent, cols):
        super().__init__(parent)
        self.setWindowTitle("Try one example")
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.edits = {}
        inner = QWidget(); inner.setLayout(form)
        for c in cols:
            e = QLineEdit(); e.setPlaceholderText(str(c.get("example") or ""))
            self.edits[c["name"]] = e
            form.addRow(f"{c['name']} ({c['kind']})", e)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(inner)
        lay.addWidget(sc)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(460, 520)

    def values(self):
        return {k: (e.text().strip() or None) for k, e in self.edits.items()}


class TryScreen(Page):
    def __init__(self, win):
        super().__init__(win, "Try your model", "Test the model on new data without writing any code: predict a whole CSV file, or one example.")
        self.model = QComboBox()
        self.root.addWidget(self.model)
        row = QHBoxLayout()
        row.addWidget(self.primary("Predict a CSV file…", self.pick_csv))
        row.addWidget(self.button("Try one example…", self.one_row))
        row.addWidget(self.button("Save predictions…", self.save))
        self.root.addLayout(row)
        self.out = lbl("")
        self.root.addWidget(self.out)
        self.table = make_table([])
        self.root.addWidget(self.table, 1)
        self.last = None

    def refresh(self):
        self.model.clear()
        ms = self.win.ws.models()
        ms.sort(key=lambda m: (m["experiment_id"] != self.win.experiment_id, not m["selected"]))
        for m in ms:
            self.model.addItem(f"{m['id']} — {m['cand_key']}{' (chosen)' if m['selected'] else ''}", m["id"])

    def pick_csv(self):
        p, _ = QFileDialog.getOpenFileName(self, "CSV with new data", "", "CSV files (*.csv *.tsv *.txt);;All files (*)")
        if p:
            self.predict_path(p)

    def predict_path(self, path):
        mid = self.model.currentData()
        if not mid:
            self.out.setText("No model yet: train one first.")
            return None
        df = self.win.guard(self.win.ws.try_model, mid, path)
        if df is not None:
            self._show(df, f"{len(df)} predictions from {Path(path).name}")
        return df

    def one_row(self):
        mid = self.model.currentData()
        if not mid:
            return
        m = self.win.ws.store.one("SELECT experiment_id FROM models WHERE id=?", mid)
        summ = self.win.ws.summary(m["experiment_id"])
        cols = [{"name": c, "kind": k} for c, k in summ["feature_kinds"].items()]
        d = RowDialog(self, cols)
        if d.exec() == QDialog.Accepted:
            self.predict_row(d.values())

    def predict_row(self, row):
        df = self.win.guard(self.win.ws.try_model, self.model.currentData(), row)
        if df is not None:
            p = df.iloc[0]
            conf = f" (confidence {p['confidence']:.0%})" if "confidence" in df else ""
            self._show(df, f"Prediction: {p['prediction']}{conf}")
        return df

    def _show(self, df, msg):
        self.last = df
        self.out.setText(msg)
        show = df.head(200)
        self.table.setColumnCount(len(show.columns)); self.table.setHorizontalHeaderLabels([str(c) for c in show.columns]); self.table.setRowCount(len(show))
        for r in range(len(show)):
            for c in range(len(show.columns)):
                v = show.iat[r, c]
                self.table.setItem(r, c, QTableWidgetItem(f"{v:.4g}" if isinstance(v, float) else str(v)))
        self.table.resizeColumnsToContents()

    def save(self):
        if self.last is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save predictions", "predictions.csv", "CSV (*.csv)")
        if p:
            self.last.to_csv(p, index=False)
            self.out.setText(f"Saved {len(self.last)} predictions to {p}")


# ------------------------------------------------------------------ improve with more data
class ImproveScreen(Page):
    def __init__(self, win):
        super().__init__(win, "Improve a model", "Pick a model you already trained, add a new dataset, and Epoch retrains on old + new data. "
                                                  "The original locked test set stays the same, so the old and new models are compared fairly.")
        self.model = QComboBox()
        self.model.currentIndexChanged.connect(self._show_hints)
        self.root.addWidget(self.model)
        self.hints = QTextEdit(); self.hints.setReadOnly(True); self.hints.setMaximumHeight(150)
        self.hints.setPlaceholderText("What would most likely improve this model appears here.")
        self.root.addWidget(lbl("What the measurements suggest for this model", "muted"))
        self.root.addWidget(self.hints)
        row = QHBoxLayout()
        row.addWidget(self.button("Choose new data (CSV)…", self.pick))
        self.run_btn = self.primary("Retrain with the new data", self.run)
        self.run_btn.setEnabled(False)
        row.addWidget(self.run_btn)
        self.cmp_btn = self.button("Compare old and new", self.compare)
        self.cmp_btn.setEnabled(False)
        row.addWidget(self.cmp_btn)
        self.root.addLayout(row)
        self.check_box = QTextEdit(); self.check_box.setReadOnly(True)
        self.root.addWidget(lbl("Compatibility and differences", "muted"))
        self.root.addWidget(self.check_box, 2)
        self.cmp_box = QTextEdit(); self.cmp_box.setReadOnly(True)
        self.root.addWidget(lbl("Comparison", "muted"))
        self.root.addWidget(self.cmp_box, 2)
        self.new_did = None

    def refresh(self):
        self.model.clear()
        ms = [m for m in self.win.ws.models() if m["selected"]]
        for m in ms:
            self.model.addItem(f"{m['id']} — {m['cand_key']} (from {m['experiment_id']})", m["id"])
        self._show_hints()

    def _show_hints(self, *_):
        mid = self.model.currentData()
        if not mid or not self.win.ws:
            return
        eid = self.win.ws.store.one("SELECT experiment_id FROM models WHERE id=?", mid)["experiment_id"]
        summ = self.win.ws.summary(eid) or {}
        recs = [r for r in summ.get("recommendations", []) if r["severity"] in ("high", "medium", "info")]
        self.hints.setPlainText("\n\n".join(f"{r['title']}\n  - " + "\n  - ".join(r["actions"]) for r in recs[:5]) or "No specific suggestions yet. More data of the same kind is the usual next step.")

    def pick(self):
        p, _ = QFileDialog.getOpenFileName(self, "New data (CSV)", "", "CSV files (*.csv *.tsv *.txt);;All files (*)")
        if p:
            self.load_new(p)

    def load_new(self, path):
        w = self.win
        mid = self.model.currentData()
        if not mid:
            return None
        w.setCursor(Qt.WaitCursor)
        try:
            did = w.guard(w.ws.import_dataset, path)
            rep = w.guard(w.ws.check_new_data, mid, did) if did else None
        finally:
            w.unsetCursor()
        if rep is None:
            return None
        self.new_did = did
        lines = [rep["summary"]] + [f"[warning] {x}" for x in rep["warnings"]] + [f"[error] {x}" for x in rep["errors"]]
        if rep["drift"]:
            lines.append("\nColumns that differ most from the original data:")
            lines += [f"  {d['column']}: {d['note']}" for d in rep["drift"][:6]]
        self.check_box.setPlainText("\n".join(lines))
        self.run_btn.setEnabled(rep["ok"])
        return rep

    def run(self):
        mid, did = self.model.currentData(), self.new_did
        if mid and did:
            self.win.start_improvement(mid, did)

    def compare(self):
        w = self.win
        old = getattr(w, "improving_from", None)
        new = w.ws.selected_model(w.experiment_id) if w.experiment_id else None
        if not (old and new):
            return
        w.setCursor(Qt.WaitCursor)
        try:
            c = w.guard(w.ws.compare_improvement, old, new)
        finally:
            w.unsetCursor()
        if not c:
            return
        k = c["metric"]
        fmt = lambda r: f"{r[k]:.4f} (n={r['n']})"
        verdict = {"adopt": "ADOPT the new model", "keep_old": "KEEP the old model", "inconclusive": "INCONCLUSIVE"}[c["verdict"]]
        self.cmp_box.setPlainText(
            f"{verdict}\n{c['explanation']}\n\nMetric: {k}\n"
            f"Original locked test set:  old {fmt(c['old_test'])}   new {fmt(c['new_test'])}   (change {c['delta_original_test']:+.4f})\n"
            f"Held-out part of the new data:  old {fmt(c['old_new_data'])}   new {fmt(c['new_new_data'])}   (change {c['delta_new_data']:+.4f})\n\n{c['note']}")
