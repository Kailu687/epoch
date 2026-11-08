import json
import time
from pathlib import Path

import pandas as pd
import pytest

from conftest import make_cls, make_noise, make_reg
from engine_trainer import inference
from engine_trainer.workspace import WorkspaceError

CANDS = ["logreg", "rf"]


def run(ws, csv, **kw):
    did = ws.import_dataset(csv)
    eid = ws.create_experiment(did, n_jobs=1, **kw)
    ws.run_blocking(eid)
    return eid, ws.experiment(eid)


def val_metric(exp, key, cand):
    return next(c for c in exp["candidates"] if c["cand_key"] == cand)["val"][key]


def test_classification_beats_baseline_and_reports(ws, csv_of):
    eid, exp = run(ws, csv_of(make_cls()), target="label", task_type="classification", candidates=CANDS)
    assert exp["status"] == "finished", exp["error"]
    assert val_metric(exp, "macro_f1", "baseline") < 0.55
    best = max(c["val"]["macro_f1"] for c in exp["candidates"])
    assert best > 0.8
    v = exp["candidates"][1]["val"]
    assert {"accuracy", "balanced_accuracy", "macro_f1", "confusion_matrix", "per_class"} <= set(v)
    assert exp["split"]["sizes"]["train"] > exp["split"]["sizes"]["val"]


def test_noise_gives_chance_level_not_inflated(ws, csv_of):
    eid, exp = run(ws, csv_of(make_noise()), target="label", task_type="classification", candidates=CANDS)
    assert exp["status"] == "finished"
    ws_mid = ws.selected_model(eid)
    rep = ws.evaluate_test(ws_mid)
    assert rep["balanced_accuracy"] < 0.65  # no real signal -> near 0.5


def test_regression(ws, csv_of):
    eid, exp = run(ws, csv_of(make_reg()), target="y", task_type="regression", candidates=CANDS)
    assert exp["status"] == "finished", exp["error"]
    base_rmse = val_metric(exp, "rmse", "baseline")
    best = min(c["val"]["rmse"] for c in exp["candidates"] if c["cand_key"] != "baseline")
    assert best < 0.4 * base_rmse
    assert ws.evaluate_test(ws.selected_model(eid))["r2"] > 0.8


def test_reproducible_split_and_metrics(ws, csv_of):
    p = csv_of(make_cls())
    _, e1 = run(ws, p, target="label", task_type="classification", candidates=["logreg"], seed=11)
    _, e2 = run(ws, p, target="label", task_type="classification", candidates=["logreg"], seed=11)
    assert e1["split"]["hash"] == e2["split"]["hash"]
    assert val_metric(e1, "macro_f1", "logreg") == val_metric(e2, "macro_f1", "logreg")


def test_test_set_is_protected(ws, csv_of):
    eid, _ = run(ws, csv_of(make_cls()), target="label", task_type="classification", candidates=["logreg"])
    mid = ws.selected_model(eid)
    ws.evaluate_test(mid)
    with pytest.raises(WorkspaceError, match="already been used"):
        ws.evaluate_test(mid)
    ws.evaluate_test(mid, acknowledge_repeat=True)
    assert ws.store.one("SELECT test_eval_count c FROM models WHERE id=?", mid)["c"] == 2
    # candidate selection used validation only: no candidate record carries test metrics before evaluate_test
    exp = ws.experiment(eid)
    assert all(c["val"] is not None for c in exp["candidates"] if c["status"] == "done")


def test_validation_error_stops_run_cleanly(ws, csv_of):
    df = make_cls(); df["label"] = "same"
    eid, exp = run(ws, csv_of(df), target="label", task_type="classification")
    assert exp["status"] == "failed" and "constant column" in exp["error"]


def test_dataset_tamper_detected(ws, csv_of):
    did = ws.import_dataset(csv_of(make_cls()))
    eid = ws.create_experiment(did, target="label", task_type="classification", candidates=["logreg"], n_jobs=1)
    with open(ws.dataset(did)["stored_path"], "a") as f:
        f.write("0,0,x,pos\n")
    ws.run_blocking(eid)
    assert "checksum mismatch" in ws.experiment(eid)["error"]


def test_export_bundle_roundtrip_and_safety(ws, csv_of, tmp_path):
    df = make_cls(id_col=True)
    eid, exp = run(ws, csv_of(df), target="label", task_type="classification", candidates=["logreg"])
    mid = ws.selected_model(eid)
    ws.evaluate_test(mid)
    out = tmp_path / "bundle"
    ws.export(mid, out)
    names = {p.name for p in out.iterdir()}
    assert {"model.joblib", "schema.json", "labels.json", "config.json", "metrics.json", "requirements.txt",
            "predict_example.py", "inference.py", "manifest.json"} <= names
    assert inference.verify_bundle(out) == []
    schema = json.loads((out / "schema.json").read_text())
    assert "customer_id" not in [c["name"] for c in schema["columns"]]
    rep = json.loads((out / "metrics.json").read_text())
    assert rep["test"] is not None and rep["test_evaluations"] == 1
    # refuses to load without explicit trust acknowledgement
    with pytest.raises(PermissionError):
        inference.load_bundle(out)
    preds = inference.predict(out, df.drop(columns=["label"]).head(50), i_trust_this_bundle=True)
    assert set(preds) <= {"pos", "neg"} and (preds == df.label.head(50).to_numpy()).mean() > 0.7
    # missing column in new data -> clear error
    with pytest.raises(ValueError, match="missing required"):
        inference.predict(out, df[["a"]].head(3), i_trust_this_bundle=True)
    # tampering is detected before anything is loaded
    (out / "schema.json").write_text("{}")
    assert any("checksum mismatch" in p for p in inference.verify_bundle(out))
    with pytest.raises(ValueError, match="integrity"):
        inference.load_bundle(out, i_trust_this_bundle=True)


def test_export_refuses_nonempty_dir(ws, csv_of, tmp_path):
    eid, _ = run(ws, csv_of(make_cls()), target="label", task_type="classification", candidates=["logreg"])
    d = tmp_path / "x"; d.mkdir(); (d / "keep.txt").write_text("hi")
    with pytest.raises(FileExistsError):
        ws.export(ws.selected_model(eid), d)


def test_cancellation_terminates_worker(ws, csv_of):
    import numpy as np
    rng = np.random.RandomState(0)
    n = 60000
    big = pd.DataFrame(rng.randn(n, 30), columns=[f"f{i}" for i in range(30)])
    big["label"] = np.where(big.f0 + rng.randn(n) > 0, "a", "b")
    did = ws.import_dataset(csv_of(big, "big.csv"))
    eid = ws.create_experiment(did, target="label", task_type="classification", n_jobs=1)
    job = ws.start(eid)
    seen = []
    t0 = time.time()
    while time.time() - t0 < 120 and not any(e["type"] == "candidate_start" for e in seen):
        seen += job.poll(0.2)
    for e in seen:
        ws.handle_event(eid, e)
    for e in job.cancel():
        ws.handle_event(eid, e)
    assert job.done and not job._proc.is_alive()
    assert ws.experiment(eid)["status"] == "cancelled"


def test_resource_events_emitted(ws, csv_of):
    did = ws.import_dataset(csv_of(make_cls()))
    eid = ws.create_experiment(did, target="label", task_type="classification", candidates=["logreg"], n_jobs=1)
    ev = ws.start(eid).wait()
    assert any(e["type"] == "resource" and e["ram_percent"] > 0 for e in ev) or any(e["type"] == "finished" for e in ev)
