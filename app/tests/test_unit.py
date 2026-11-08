import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from conftest import make_cls, make_reg
from engine_trainer import inference
from engine_trainer.io_csv import DatasetError, read_csv
from engine_trainer.pipeline import build_preprocessor, prepare_frame
from engine_trainer.profile import infer_task, profile_columns
from engine_trainer.splits import SplitError, make_split, split_hash
from engine_trainer.validate import leakage_probe, validate


# ------------------------------------------------------------------ io
def test_semicolon_and_latin1(tmp_path):
    p = tmp_path / "s.csv"
    p.write_bytes("name;val;label\n".encode() + b"".join(f"caf\xe9{i};{i};{i % 2}\n".encode("latin-1") for i in range(20)))
    df, info = read_csv(p)
    assert df.shape == (20, 3) and info.delimiter == ";" and df.name[0].startswith("caf")


@pytest.mark.parametrize("content,msg", [(b"", "empty"), (b"a,b\n1,2\n", "rows"), (b"\x00\x01\x02" * 100, "binary"),
                                          (b"a,a\n" + b"1,2\n" * 20, "Duplicate")])
def test_bad_files_rejected(tmp_path, content, msg):
    p = tmp_path / "bad.csv"
    p.write_bytes(content)
    with pytest.raises(DatasetError, match=msg):
        read_csv(p)


def test_missing_file():
    with pytest.raises(DatasetError):
        read_csv("does_not_exist.csv")


# ------------------------------------------------------------------ profile
def test_column_kinds():
    df = make_cls(id_col=True)
    df["const"] = 1
    df["when"] = pd.date_range("2020-01-01", periods=len(df)).astype(str)
    kinds = {p.name: p.kind for p in profile_columns(df)}
    assert kinds["customer_id"] == "id" and kinds["const"] == "constant" and kinds["when"] == "datetime"
    assert kinds["a"] == "numeric" and kinds["cat"] == "categorical"


def test_task_inference():
    assert infer_task(make_cls(), "label").task_type == "classification"
    assert infer_task(make_reg(), "y").task_type == "regression"
    df = pd.DataFrame({"t": np.random.RandomState(0).randint(0, 3, 200), "x": range(200)})
    s = infer_task(df, "t")
    assert s.task_type == "classification" and s.needs_confirmation  # ambiguous integer target must ask


# ------------------------------------------------------------------ splits
def test_split_deterministic_disjoint_stratified():
    y = np.array(["a"] * 160 + ["b"] * 40)
    s1, s2 = make_split(y, "classification", seed=7), make_split(y, "classification", seed=7)
    assert split_hash(s1) == split_hash(s2)
    assert split_hash(s1) != split_hash(make_split(y, "classification", seed=8))
    all_idx = np.concatenate([s1["train"], s1["val"], s1["test"]])
    assert sorted(all_idx) == list(range(200))  # disjoint and complete
    for k in ("train", "val", "test"):
        frac = (y[s1[k]] == "b").mean()
        assert abs(frac - 0.2) < 0.03  # stratified


def test_rare_class_rejected():
    y = np.array(["a"] * 50 + ["b"] * 2)
    with pytest.raises(SplitError):
        make_split(y, "classification")


# ------------------------------------------------------------------ validation
def test_validation_findings():
    df = make_cls(id_col=True, planted_leak=True)
    vr = validate(df, profile_columns(df), "label", "classification")
    codes = {f.code for f in vr.findings}
    assert vr.ok and "customer_id" not in vr.features and "excluded_id" in codes


def test_leakage_probe_flags_planted_column_only():
    df = make_cls(planted_leak=True)
    vr = validate(df, profile_columns(df), "label", "classification")
    X = prepare_frame(df, vr.feature_kinds)
    tr = make_split(df.label.to_numpy(), "classification")["train"]
    flagged = {f.column for f in leakage_probe(X.iloc[tr], df.label.to_numpy()[tr], vr.feature_kinds, "classification")}
    assert flagged == {"post_outcome"}


def test_validation_errors():
    df = make_cls()
    df["label"] = "one"
    assert not validate(df, profile_columns(df), "label", "classification").ok
    df2 = make_cls(); df2["label"] = ["pos"] * 498 + ["rare", "rare"]
    assert "rare_classes" in {f.code for f in validate(df2, profile_columns(df2), "label", "classification").errors}
    r = make_reg(); r["y"] = r["g"]
    assert not validate(r, profile_columns(r), "y", "regression").ok


def test_imbalance_warning():
    df = make_cls(); df["label"] = ["pos"] * 450 + ["neg"] * 50
    assert "imbalance" in {f.code for f in validate(df, profile_columns(df), "label", "classification").findings}


# ------------------------------------------------------------------ preprocessing never sees held-out data
def test_scaler_fit_on_train_only():
    df = make_cls()
    kinds = {"a": "numeric", "b": "numeric", "cat": "categorical"}
    X = prepare_frame(df, kinds)
    tr = np.arange(0, 300)
    prep = build_preprocessor(kinds).fit(X.iloc[tr])
    scaler: StandardScaler = prep.named_transformers_["num"].named_steps["scale"]
    train_mean_b = X["b"].iloc[tr].mean()
    assert np.isclose(scaler.mean_[1], train_mean_b)
    assert not np.isclose(scaler.mean_[1], X["b"].mean())


def test_prepare_input_schema_error():
    schema = {"columns": [{"name": "a", "kind": "numeric"}, {"name": "cat", "kind": "categorical"}]}
    with pytest.raises(ValueError, match="missing required"):
        inference.prepare_input(pd.DataFrame({"a": [1.0]}), schema)
