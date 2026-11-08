import os

if not os.environ.get("ET_NATIVE_QT"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["ET_SETTINGS_ORG"] = "EpochTests"

from PySide6.QtWidgets import QApplication

from engine_trainer.gui.app import STYLE, MainWindow
from test_gui import pump, set_combo, shot  # noqa: F401  (shared helpers)


def test_let_epoch_decide_on_the_demo_dataset(tmp_path):
    """A non-ML user: demo data, a plain-language description, no manual choices, one click to train."""
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    assert w.open_workspace(str(tmp_path / "ws"))
    w.go("import")
    shot(w, "10_import_sample")
    w.screens["import"].make_sample()
    pump(lambda: w.dataset_id)
    assert (tmp_path / "ws" / "samples" / "synthetic_churn_demo.csv").is_file()
    t = w.screens["task"]
    w.go("task")
    t.desc.setPlainText("Customer records. I want to know who is likely to leave.")
    t.ask()
    shot(w, "10b_task_advice")
    # Epoch understood the goal without any manual choice
    assert t.target.currentData() == "churned" and t.task.currentData() == "classification"
    assert "customer_id" in w.cfg["exclude"] and w.cfg["advice"] is not None
    assert t.next.isEnabled()
    w.go("config")
    cf = w.screens["config"]
    cf.seconds.setValue(30); cf.jobs.setValue(2)
    cf.go_train()
    pump(lambda: w.job.done)
    w.go("results")
    r = w.screens["results"]
    names = [n for n, _ in r.chart.rows]
    assert "baseline" in names and len(names) >= 3 and all(v >= 0 for _, v in r.chart.rows)
    assert "advice" in (w.ws.summary(w.experiment_id) or {})
    shot(w, "11_results_chart")
    w.close()
