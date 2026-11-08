"""Drive the real GUI offscreen through the full AutoML workflow (no mocks of the engine)."""
import os
import time
from pathlib import Path

if not os.environ.get("ET_NATIVE_QT"):  # set ET_NATIVE_QT=1 to render with the real Windows platform (fonts) for screenshots
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QApplication

from conftest import make_cls
from engine_trainer import inference
from engine_trainer.gui.app import STYLE, MainWindow

os.environ["ET_SETTINGS_ORG"] = "EpochTests"
SHOTS = os.environ.get("ET_SHOTS")


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(STYLE)
    return a


def pump(cond, timeout=180):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        QCoreApplication.processEvents()
        time.sleep(0.02)
    assert cond(), "timed out"


def shot(w, name):
    if SHOTS:
        Path(SHOTS).mkdir(exist_ok=True, parents=True)
        w.grab().save(str(Path(SHOTS) / f"{name}.png"))


def set_combo(combo, data):
    i = combo.findData(data)
    assert i >= 0, data
    combo.setCurrentIndex(i)


def test_full_gui_workflow(app, tmp_path):
    csv = tmp_path / "customers.csv"
    make_cls(id_col=True, planted_leak=True).to_csv(csv, index=False)
    w = MainWindow()
    w.resize(1100, 760)
    w.show()
    shot(w, "01_welcome")
    assert not (w.nav.item(2).flags() & Qt.ItemIsEnabled)  # nav is gated before a workspace exists
    assert w.open_workspace(str(tmp_path / "ws"))
    w.go("import"); shot(w, "02_import")
    w.screens["import"].do_import(str(csv))
    assert w.screens["import"].msg.text().startswith("Reading")  # returned immediately: work happens off the UI thread
    pump(lambda: w.dataset_id)
    shot(w, "03_explorer")
    assert w.screens["explorer"].preview.rowCount() == 50

    w.go("task")
    t = w.screens["task"]
    t.desc.setPlainText("I want to predict the label.")
    t.ask()
    assert t.advice_box.toPlainText()  # Epoch explains its choices in words
    set_combo(t.target, "label")  # a manual choice always wins
    assert t.task.currentData() == "classification"
    t.validate(); shot(w, "04_task")
    assert t.next.isEnabled()
    texts = [t.findings.item(i).text() for i in range(t.findings.count())]
    assert "customer_id" in w.cfg["exclude"] or any("customer_id" in x for x in texts)  # the Advisor drops id columns before validation

    w.go("models")
    ms = w.screens["models"]
    for i in range(ms.list.count()):  # keep it fast: baseline + linear only
        it = ms.list.item(i)
        if it.data(Qt.UserRole) not in ("baseline", "linear"):
            it.setCheckState(Qt.Unchecked)
    assert w.cfg["models"] == ["baseline", "linear"]
    shot(w, "04b_models")
    w.go("config")
    cf = w.screens["config"]
    set_combo(cf.effort, "express"); cf.seconds.setValue(30); cf.jobs.setValue(2)
    shot(w, "04c_config")
    cf.go_train()
    assert w.job is not None
    pump(lambda: w.job.done)
    shot(w, "05_monitor")
    mon = w.screens["monitor"]
    assert mon.table.rowCount() >= 2 and mon.results.isEnabled()

    w.go("results")
    r = w.screens["results"]
    assert r.summary.text().startswith("Chosen:") and "baseline" in r.summary.text().lower()
    assert r.recs.toPlainText()
    r.eval_test(); shot(w, "06_results")
    assert "evaluated 1 time" in r.test_out.toPlainText()

    w.go("try")
    ty = w.screens["try"]
    assert ty.model.count() == 2
    test_csv = tmp_path / "new.csv"
    make_cls(id_col=True, planted_leak=True).drop(columns=["label"]).head(20).to_csv(test_csv, index=False)
    df = ty.predict_path(str(test_csv))
    assert len(df) == 20 and {"prediction", "confidence"} <= set(df.columns)
    one = ty.predict_row({"a": 0.5, "b": -0.2, "cat": "x", "post_outcome": 1.0})
    assert one is not None and len(one) == 1
    shot(w, "06d_try")

    w.go("history"); assert w.screens["history"].table.rowCount() == 1
    w.go("registry"); assert w.screens["registry"].table.rowCount() == 2
    w.go("export")
    e = w.screens["export"]
    e.do_export(str(tmp_path / "out"))
    shot(w, "07_export")
    assert "Exported" in e.out.text()
    assert inference.verify_bundle(tmp_path / "out") == []
    assert (tmp_path / "out" / "report.html").is_file()
    w.go("settings"); shot(w, "08_settings")
    assert "Epoch" in w.screens["settings"].info.toPlainText()
    w.close()


def test_gui_import_error_is_reported_not_crashing(app, tmp_path, monkeypatch):
    bad = tmp_path / "bad.csv"
    bad.write_text("only_one_column\n1\n2\n")
    w = MainWindow()
    shown = []
    monkeypatch.setattr(MainWindow, "error", lambda self, m: shown.append(m))
    w.open_workspace(str(tmp_path / "ws"))
    w.screens["import"].do_import(str(bad))
    pump(lambda: shown)
    assert w.dataset_id is None and w.screens["import"].import_btn.isEnabled()
    w.close()


def test_gui_cancel(app, tmp_path):
    import numpy as np, pandas as pd
    rng = np.random.RandomState(0)
    n = 60000
    df = pd.DataFrame(rng.randn(n, 30), columns=[f"f{i}" for i in range(30)])
    df["label"] = np.where(df.f0 + rng.randn(n) > 0, "a", "b")
    csv = tmp_path / "big.csv"; df.to_csv(csv, index=False)
    w = MainWindow()
    w.open_workspace(str(tmp_path / "ws"))
    w.screens["import"].do_import(str(csv))
    pump(lambda: w.dataset_id)
    w.cfg.update(target="label", task_type="classification", n_jobs=1, budget_seconds=300)
    w.start_training()
    pump(lambda: w.screens["monitor"].table.rowCount() > 0)
    w.cancel_training()
    pump(lambda: w.job.done)
    assert w.ws.experiment(w.experiment_id)["status"] == "cancelled"
    assert w.screens["monitor"].stage.text() == "Cancelled."
    w.close()
