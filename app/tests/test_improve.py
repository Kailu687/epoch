"""Improve an existing model with more data; reports and Try-model helpers."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_cls
from engine_trainer.improve import check_compatibility
from engine_trainer.workspace import WorkspaceError

FAST = dict(mode="automl", budget_seconds=30, max_trials=2, n_jobs=2, models=["linear", "rf"])


def train(ws, df, csv_of, name):
    did = ws.import_dataset(csv_of(df, name))
    eid = ws.create_experiment(did, target="label", task_type="classification", **FAST)
    ws.run_blocking(eid)
    assert ws.experiment(eid)["status"] == "finished", ws.experiment(eid)["error"]
    return did, eid


def test_compatibility_and_drift_detection():
    old = make_cls(n=400, seed=0)
    kinds = {"a": "numeric", "b": "numeric", "cat": "categorical"}
    ok = check_compatibility(old, make_cls(n=200, seed=5), kinds, "label", "classification", ["neg", "pos"])
    assert ok["ok"] and not any("different" in w for w in ok["warnings"])
    missing = check_compatibility(old, make_cls(n=200, seed=5).drop(columns=["b"]), kinds, "label", "classification", ["neg", "pos"])
    assert not missing["ok"] and "b" in missing["errors"][0]
    shifted = make_cls(n=200, seed=5)
    shifted["a"] = shifted["a"] * 3 + 10
    shifted["cat"] = "novel"
    rep = check_compatibility(old, shifted, kinds, "label", "classification", ["neg", "pos"])
    assert rep["ok"] and rep["drift"][0]["column"] in ("a", "cat") and any("different" in w for w in rep["warnings"])
    new_cls = make_cls(n=200, seed=5)
    new_cls.loc[:10, "label"] = "maybe"
    assert any("not seen" in w for w in check_compatibility(old, new_cls, kinds, "label", "classification", ["neg", "pos"])["warnings"])


def test_improve_keeps_the_same_locked_test_set(ws, csv_of):
    did, eid = train(ws, make_cls(n=500, seed=0), csv_of, "old.csv")
    old_mid = ws.selected_model(eid)
    ws.evaluate_test(old_mid)
    new_did = ws.import_dataset(csv_of(make_cls(n=300, seed=7), "new.csv"))
    rep = ws.check_new_data(old_mid, new_did)
    assert rep["ok"]
    eid2 = ws.improve_model(old_mid, new_did)
    ws.run_blocking(eid2)
    exp2 = ws.experiment(eid2)
    assert exp2["status"] == "finished", exp2["error"]
    old_sp = json.loads((Path(ws.experiment(eid)["run_dir"]) / "split.json").read_text())
    new_sp = json.loads((Path(exp2["run_dir"]) / "split.json").read_text())
    old_test = sorted(old_sp["original_rows"][i] for i in old_sp["test"])
    new_test = sorted(new_sp["original_rows"][i] for i in new_sp["test"])
    assert old_test == new_test  # identical rows: the comparison is fair
    hold = {new_sp["original_rows"][i] for i in new_sp["new_holdout"]}
    assert hold and min(hold) >= 500  # held out rows come from the new data only
    dev = {new_sp["original_rows"][i] for i in new_sp["dev"]}
    assert not dev & set(old_test) and not dev & hold  # nothing leaks into the test sets
    cmp = ws.compare_improvement(old_mid, ws.selected_model(eid2))
    assert cmp["verdict"] in ("adopt", "keep_old", "inconclusive") and cmp["explanation"]
    assert cmp["old_test"]["n"] == cmp["new_test"]["n"] == len(old_test)


def test_incompatible_new_data_is_refused(ws, csv_of):
    did, eid = train(ws, make_cls(n=400, seed=0), csv_of, "old.csv")
    bad = ws.import_dataset(csv_of(make_cls(n=200, seed=3).drop(columns=["a"]), "bad.csv"))
    with pytest.raises(WorkspaceError, match="missing"):
        ws.improve_model(ws.selected_model(eid), bad)


def test_try_model_and_report(ws, csv_of):
    df = make_cls(n=500, seed=0)
    did, eid = train(ws, df, csv_of, "d.csv")
    mid = ws.selected_model(eid)
    out = ws.try_model(mid, df.drop(columns=["label"]).head(25))
    assert len(out) == 25 and {"prediction", "confidence"} <= set(out.columns)
    assert (out["prediction"].to_numpy() == df.label.head(25).to_numpy()).mean() > 0.7
    one = ws.try_model(mid, {"a": 0.1, "b": 0.2, "cat": "x"})
    assert len(one) == 1
    with pytest.raises(WorkspaceError, match="missing required"):
        ws.try_model(mid, pd.DataFrame({"a": [1.0]}))
    html = ws.report_html(mid)
    assert "Epoch evaluation report" in html and "Recommendations" in html and "Locked test set" in html
    assert "not evaluated yet" in html  # honest: the test set has not been used
    ws.evaluate_test(mid)
    assert "not evaluated yet" not in ws.report_html(mid)
