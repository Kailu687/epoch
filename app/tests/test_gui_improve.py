import os

if not os.environ.get("ET_NATIVE_QT"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["ET_SETTINGS_ORG"] = "EpochTests"

from PySide6.QtWidgets import QApplication

from conftest import make_cls
from engine_trainer.gui.app import STYLE, MainWindow
from test_gui import pump, set_combo, shot  # noqa: F401


def test_improve_a_model_with_new_data_in_the_gui(tmp_path):
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLE)
    make_cls(n=450, seed=0).to_csv(tmp_path / "old.csv", index=False)
    make_cls(n=300, seed=9).to_csv(tmp_path / "new.csv", index=False)
    w = MainWindow()
    w.show()
    assert w.open_workspace(str(tmp_path / "ws"))
    did = w.ws.import_dataset(str(tmp_path / "old.csv"))
    w.set_dataset(did)
    w.cfg.update(target="label", task_type="classification", n_jobs=2, budget_seconds=25, models=["linear", "rf"], max_trials=None)
    w.start_training()
    pump(lambda: w.job.done)
    first = w.experiment_id
    assert w.ws.experiment(first)["status"] == "finished"
    w.go("improve")
    im = w.screens["improve"]
    assert im.model.count() == 1
    rep = im.load_new(str(tmp_path / "new.csv"))
    shot(w, "12_improve")
    assert rep and rep["ok"] and im.run_btn.isEnabled()
    im.run()
    pump(lambda: w.job.done)
    assert w.experiment_id != first and w.ws.experiment(w.experiment_id)["status"] == "finished"
    w.go("improve")
    im.compare()
    shot(w, "12b_improve_compare")
    text = im.cmp_box.toPlainText()
    assert "Original locked test set" in text and ("ADOPT" in text or "KEEP" in text or "INCONCLUSIVE" in text)
    w.close()
