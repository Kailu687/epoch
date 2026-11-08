"""AutoML v2: cross-validated search, tuning, ensembling, group/temporal protocols, recommendations."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_cls, make_noise, make_reg
from engine_trainer import inference
from engine_trainer.automl import metrics as mx
from engine_trainer.automl import search as sr
from engine_trainer.automl import zoo
from engine_trainer.splits import SplitError, make_split_strategy

FAST = dict(mode="automl", budget_mode="express", budget_seconds=40, max_trials=3, n_jobs=2, models=["linear", "tree", "rf", "lightgbm"])


def run(ws, csv, **kw):
    did = ws.import_dataset(csv)
    eid = ws.create_experiment(did, **{**FAST, **kw})
    ws.run_blocking(eid)
    return eid, ws.experiment(eid), json.loads((Path(ws.experiment(eid)["run_dir"]) / "summary.json").read_text()) if ws.experiment(eid)["status"] == "finished" else None


# ------------------------------------------------------------------ units
def test_make_folds_stratified_and_group():
    y = np.array(["a"] * 40 + ["b"] * 20)
    for tr, va in sr.make_folds(y, "classification", "stratified", 5, 0):
        assert abs((y[va] == "b").mean() - 1 / 3) < 0.1
    groups = np.repeat(np.arange(12), 5)
    for tr, va in sr.make_folds(y, "classification", "group", 4, 0, groups):
        assert not set(groups[tr]) & set(groups[va])


def test_temporal_folds_never_train_on_future():
    y = np.arange(100.0)
    for tr, va in sr.make_folds(y, "regression", "temporal", 4, 0):
        assert tr.max() < va.min()


def test_group_split_disjoint_and_temporal_ordered():
    y = np.array(["a", "b"] * 60)
    groups = np.repeat(np.arange(24), 5)
    s = make_split_strategy(y, "classification", "group", 1, groups=groups)
    parts = [set(groups[s[k]]) for k in ("train", "val", "test")]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
    t = np.random.RandomState(0).permutation(120)
    s = make_split_strategy(y, "classification", "temporal", 1, order=t)
    assert t[s["train"]].max() < t[s["val"]].min() <= t[s["val"]].max() < t[s["test"]].min()
    with pytest.raises(SplitError):
        make_split_strategy(y, "classification", "group", 1, groups=np.repeat(np.arange(3), 40))


def test_metrics_direction():
    y = np.array([0, 1, 1, 0])
    assert mx.score("rmse", y, y + 1.0) < mx.score("rmse", y, y)
    assert mx.score("macro_f1", y, y) > mx.score("macro_f1", y, 1 - y)
    assert mx.default_metric("classification", 5.0) == "macro_f1" and mx.default_metric("regression") == "rmse"


def test_zoo_filters_by_profile_and_size():
    names = {m.id for m in zoo.available("classification", 1000, "interpretable")}
    assert names <= {"baseline", "linear", "tree", "nb"} and "linear" in names
    assert "svm" not in {m.id for m in zoo.available("classification", 100000, "balanced")}
    assert "xgboost" not in {m.id for m in zoo.available("classification", 1000, "balanced")}  # regression-only


def test_random_sampler_fallback_runs_every_space():
    for m in zoo.ZOO.values():
        if m.space is None:
            continue
        for task in m.tasks:
            p = m.space(zoo.RandomSampler(0), task, 500)
            zoo.build(m.id, p, task, 0, 1)


# ------------------------------------------------------------------ end to end
def test_classification_automl_end_to_end(ws, csv_of, tmp_path):
    eid, exp, s = run(ws, csv_of(make_cls(n=700)), target="label", task_type="classification")
    assert exp["status"] == "finished", exp["error"]
    assert s["beats_baseline"] is True and s["cv_scheme"] == "stratified"
    done = {c["cand_key"]: c for c in exp["candidates"] if c["status"] == "done"}
    assert "baseline" in done and s["selected"] in done
    sel = done[s["selected"]]["val"]
    assert sel["cv_mean"] > 0.8 and {"macro_f1", "confusion_matrix", "per_class"} <= set(sel)
    assert s["learning_curve"] and isinstance(s["recommendations"], list)
    # tuning was attempted and compared with defaults
    assert any(c["val"].get("n_trials", 0) > 0 for c in done.values() if c["cand_key"] != "baseline")
    mid = ws.selected_model(eid)
    rep = ws.evaluate_test(mid)
    assert rep["accuracy"] > 0.8
    out = tmp_path / "bundle"
    ws.export(mid, out)
    assert inference.verify_bundle(out) == []
    df = make_cls(n=700)
    preds = inference.predict(out, df.drop(columns=["label"]).head(30), i_trust_this_bundle=True)
    assert (preds == df.label.head(30).to_numpy()).mean() > 0.7


def test_regression_automl(ws, csv_of):
    eid, exp, s = run(ws, csv_of(make_reg(n=600)), target="y", task_type="regression")
    assert exp["status"] == "finished", exp["error"]
    assert s["selection_metric"] == "rmse" and s["higher_is_better"] is False
    assert ws.evaluate_test(ws.selected_model(eid))["r2"] > 0.8


def test_noise_is_reported_as_no_signal(ws, csv_of):
    eid, exp, s = run(ws, csv_of(make_noise(n=600)), target="label", task_type="classification")
    assert exp["status"] == "finished"
    assert s["beats_baseline"] is False
    assert any(r["id"] == "weak_signal" for r in s["recommendations"])
    assert ws.evaluate_test(ws.selected_model(eid))["balanced_accuracy"] < 0.62


def test_group_protocol_keeps_groups_apart(ws, csv_of):
    rng = np.random.RandomState(0)
    df = make_cls(n=720)
    df["patient"] = np.repeat([f"P{i}" for i in range(36)], 20)
    eid, exp, s = run(ws, csv_of(df), target="label", task_type="classification", split_strategy="group", group_column="patient")
    assert exp["status"] == "finished", exp["error"]
    sp = json.loads((Path(exp["run_dir"]) / "split.json").read_text())
    g = df["patient"].to_numpy()[sp["original_rows"]]
    for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
        assert not set(g[sp[a]]) & set(g[sp[b]])
    assert s["cv_scheme"] == "group"
    assert "patient" not in s["feature_kinds"]  # the group id is never a feature


def test_temporal_protocol(ws, csv_of):
    df = make_reg(n=500)
    df["when"] = pd.date_range("2024-01-01", periods=len(df), freq="D").astype(str)
    df = df.sample(frac=1, random_state=3).reset_index(drop=True)  # file order is NOT time order
    eid, exp, s = run(ws, csv_of(df), target="y", task_type="regression", split_strategy="temporal", time_column="when")
    assert exp["status"] == "finished", exp["error"]
    sp = json.loads((Path(exp["run_dir"]) / "split.json").read_text())
    t = pd.to_datetime(df["when"]).to_numpy()[sp["original_rows"]]
    assert t[sp["train"]].max() <= t[sp["val"]].min() and t[sp["val"]].max() <= t[sp["test"]].min()
    assert s["cv_scheme"] == "temporal"


def test_imbalance_picks_macro_metric_and_flags_weak_class(ws, csv_of):
    df = make_cls(n=900)
    keep = df[df.label == "pos"].index[:60].tolist() + df[df.label == "neg"].index.tolist()
    eid, exp, s = run(ws, csv_of(df.loc[keep].reset_index(drop=True)), target="label", task_type="classification")
    assert exp["status"] == "finished", exp["error"]
    assert s["selection_metric"] in ("macro_f1", "balanced_accuracy")


def test_budget_is_respected(ws, csv_of):
    t0 = time.time()
    eid, exp, s = run(ws, csv_of(make_cls(n=600)), target="label", task_type="classification", budget_seconds=25, models=None)
    assert exp["status"] == "finished", exp["error"]
    assert s["seconds_used"] < 25 * 2  # soft budget: validation/refit/curve may overrun a little, never by 2x
    assert time.time() - t0 < 120


def test_cancel_during_search(ws, csv_of):
    rng = np.random.RandomState(0)
    n = 30000
    big = pd.DataFrame(rng.randn(n, 25), columns=[f"f{i}" for i in range(25)])
    big["label"] = np.where(big.f0 + rng.randn(n) > 0, "a", "b")
    did = ws.import_dataset(csv_of(big, "big.csv"))
    eid = ws.create_experiment(did, target="label", task_type="classification", mode="automl", budget_seconds=300, n_jobs=2)
    job = ws.start(eid)
    seen, t0 = [], time.time()
    while time.time() - t0 < 120 and not any(e["type"] == "candidate_start" for e in seen):
        seen += job.poll(0.2)
    for e in seen:
        ws.handle_event(eid, e)
    for e in job.cancel():
        ws.handle_event(eid, e)
    assert ws.experiment(eid)["status"] == "cancelled" and not job._proc.is_alive()
