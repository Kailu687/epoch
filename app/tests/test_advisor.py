"""Tests for the Epoch Advisor (synthetic data only; deterministic)."""
import json

import numpy as np
import pandas as pd
import pytest

from engine_trainer.advisor import Advice, advise, apply_answers, describe_advice
from engine_trainer.profile import profile_dataset

RNG = np.random.default_rng(0)


def run(desc, df, hints=None, with_df=True):
    return advise(desc, profile_dataset(df), df if with_df else None, hints)


def qids(a):
    return [q["id"] for q in a.questions]


@pytest.fixture
def churn():
    n = 400
    df = pd.DataFrame({
        "customer_id": np.arange(n),
        "tenure_months": RNG.integers(1, 72, n),
        "monthly_charges": RNG.normal(70, 20, n).round(2),
        "contract": RNG.choice(["monthly", "yearly", "two_year"], n),
        "support_calls": RNG.integers(0, 9, n),
    })
    df["churn"] = np.where(RNG.random(n) < 0.3, "yes", "no")
    return df


@pytest.fixture
def houses():
    n = 300
    area = RNG.uniform(40, 250, n)
    df = pd.DataFrame({"area": area.round(1), "rooms": RNG.integers(1, 7, n), "age_years": RNG.integers(0, 80, n),
                       "neighbourhood": RNG.choice(list("ABCDE"), n)})
    df["price"] = (area * 1800 + RNG.normal(0, 15000, n)).round(2)
    return df


def test_churn_classification(churn):
    a = run("Customer records. I want to predict who will churn.", churn)
    assert (a.task_type, a.target) == ("classification", "churn")
    assert a.split_strategy == "stratified" and a.metric in ("roc_auc", "balanced_accuracy")
    assert "customer_id" in a.exclude_columns and a.confidence == "high" and not a.questions
    assert any("customer_id" in e for e in a.explanations)


def test_churn_synonym_wording(churn):
    a = run("customer records, I want to know who will leave", churn)
    assert a.target == "churn" and a.task_type == "classification"


def test_house_price_regression_and_typo(houses):
    a = run("I want to predict the price of a house", houses)
    assert (a.task_type, a.target, a.metric) == ("regression", "price", "rmse")
    assert a.split_strategy == "random" and a.confidence == "high"
    b = run("i want to prdict the prise of houses", houses)
    assert b.target == "price"  # misspelled mention still found


def test_misspelled_column_picks_right_one():
    df = pd.DataFrame({"monthly_income": RNG.normal(3000, 500, 200).round(1), "age": RNG.integers(18, 80, 200),
                       "region": RNG.choice(list("abc"), 200), "spend": RNG.normal(100, 30, 200).round(2)})
    a = run("estimate the monthly incomee", df)
    assert a.target == "monthly_income" and a.task_type == "regression"


def test_fraud_imbalance_uses_pr_auc():
    n = 2000
    df = pd.DataFrame({"amount": RNG.exponential(50, n), "hour": RNG.integers(0, 24, n), "country": RNG.choice(["a", "b", "c"], n)})
    df["is_fraud"] = (RNG.random(n) < 0.02).astype(int)
    a = run("Detect fraud in card payments", df)
    assert (a.target, a.task_type, a.metric) == ("is_fraud", "classification", "pr_auc")
    assert any("rare" in e for e in a.explanations)


def test_moderate_imbalance_balanced_accuracy():
    n = 1000
    df = pd.DataFrame({"x1": RNG.normal(size=n), "x2": RNG.normal(size=n)})
    df["target"] = np.where(RNG.random(n) < 0.18, "bad", "good")
    assert run("", df).metric == "balanced_accuracy"


def test_multiclass_macro_f1():
    n = 600
    df = pd.DataFrame({"a": RNG.normal(size=n), "b": RNG.normal(size=n), "species": RNG.choice(["x", "y", "z"], n)})
    assert run("classify the species", df).metric == "macro_f1"


def test_heavy_outliers_use_mae():
    n = 500
    y = RNG.normal(100, 5, n)
    y[:30] = 5000
    df = pd.DataFrame({"f1": RNG.normal(size=n), "f2": RNG.normal(size=n), "price": y})
    assert run("predict price", df).metric == "mae"


def test_repeated_patient_group_split():
    n = 300
    df = pd.DataFrame({"patient_id": np.repeat(np.arange(30), 10), "heart_rate": RNG.normal(70, 8, n),
                       "bp": RNG.normal(120, 10, n)})
    df["diagnosis"] = np.where(RNG.random(n) < 0.4, "sick", "healthy")
    a = run("Each patient has several visits and I want to predict the diagnosis", df)
    assert a.split_strategy == "group" and a.group_column == "patient_id" and a.target == "diagnosis"
    assert "patient_id" in a.exclude_columns and a.confidence == "high"


def test_group_proposal_without_text_asks():
    n = 300
    df = pd.DataFrame({"speaker": np.repeat(np.arange(30), 10), "f1": RNG.normal(size=n), "f2": RNG.normal(size=n)})
    df["label"] = RNG.choice(["a", "b"], n)
    a = run("", df)
    assert a.split_strategy == "group" and "group_column" in qids(a) and a.confidence == "medium"
    b = apply_answers(a, {"group_column": "none"})
    assert b.split_strategy == "stratified" and "group_column" not in qids(b) and b.group_column is None


def test_group_wording_without_group_column_warns():
    df = pd.DataFrame({"f1": RNG.normal(size=200), "f2": RNG.normal(size=200), "label": RNG.choice(["a", "b"], 200)})
    a = run("the same patient appears in many rows", df)
    assert a.split_strategy == "stratified" and any("same person" in w for w in a.warnings)
    assert a.confidence == "medium"


def test_temporal_forecast():
    n = 120
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="D").astype(str), "store_visits": RNG.integers(50, 150, n),
                       "promo": RNG.integers(0, 2, n)})
    df["sales"] = (df["store_visits"] * 3 + RNG.normal(0, 10, n)).round(1)
    a = run("Daily sales. I want to forecast sales for next month", df)
    assert a.split_strategy == "temporal" and a.time_column == "date" and a.target == "sales" and a.task_type == "regression"
    assert "date" in a.exclude_columns


def test_monotonic_numeric_time_column():
    n = 100
    df = pd.DataFrame({"year": np.repeat(np.arange(2000, 2010), 10), "x": RNG.normal(size=n), "y_val": RNG.normal(size=n)})
    df["revenue"] = df["x"] * 10 + 50 + RNG.normal(size=n)
    a = run("forecast revenue next year", df)
    assert a.time_column == "year" and a.split_strategy == "temporal"


def test_sorted_dates_without_wording_asks():
    n = 100
    df = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=n).astype(str), "x": RNG.normal(size=n)})
    df["price"] = df["x"] * 5 + RNG.normal(size=n)
    a = run("predict the price", df)
    assert "time_column" in qids(a) and a.split_strategy == "random"
    b = apply_answers(a, {"time_column": "date"})
    assert b.split_strategy == "temporal" and "time_column" not in qids(b)


def test_forecast_wording_without_time_column_warns(houses):
    a = run("forecast the price next month", houses)
    assert a.split_strategy == "random" and any("time" in w.lower() for w in a.warnings)


def test_empty_description_data_driven(churn):
    a = run("", churn)
    assert a.target == "churn" and a.confidence in ("medium", "high")
    assert a.confidence == "medium" and any("did not describe" in e for e in a.explanations)


@pytest.mark.parametrize("text", ["I don't know", "no idea, you decide", None])
def test_unsure_description(churn, text):
    a = run(text, churn)
    assert a.target == "churn" and a.confidence == "medium"


def test_no_silent_last_column_pick():
    df = pd.DataFrame({"a": RNG.normal(size=100), "b": RNG.normal(size=100), "c": RNG.normal(size=100), "d": RNG.normal(size=100)})
    a = run("", df)
    assert a.target is None and a.task_type == "unknown" and a.metric is None and a.confidence == "low"
    assert "target" in qids(a) and len(a.target_candidates) >= 1
    b = apply_answers(a, {"target": "d"})
    assert b.target == "d" and b.task_type == "regression" and b.metric == "rmse"


def test_ambiguous_integer_target_asks():
    n = 300
    df = pd.DataFrame({"length": RNG.normal(size=n), "price_paid": RNG.normal(size=n), "stars": RNG.integers(1, 6, n)})
    a = run("I want to predict stars", df)
    assert a.target == "stars" and a.task_type == "unknown" and "task_type" in qids(a) and a.confidence == "low"
    assert a.metric is None
    b = apply_answers(a, {"task_type": "classification"})
    assert b.task_type == "classification" and not b.questions and b.metric == "macro_f1"
    c = apply_answers(a, {"task_type": "regression"})
    assert c.task_type == "regression"


def test_integer_target_resolved_by_wording():
    n = 300
    df = pd.DataFrame({"f": RNG.normal(size=n), "stars": RNG.integers(1, 6, n)})
    a = run("classify the stars category", df)
    assert a.task_type == "classification" and not a.questions


def test_many_classes_asks():
    n = 600
    df = pd.DataFrame({"f": RNG.normal(size=n), "g": RNG.normal(size=n), "city": RNG.choice([f"c{i}" for i in range(80)], n)})
    a = run("predict the city", df)
    assert a.target == "city" and "confirm_many_classes" in qids(a)
    assert apply_answers(a, {"confirm_many_classes": "keep"}).questions == []
    b = apply_answers(a, {"confirm_many_classes": "choose_another"})
    assert b.target != "city"


def test_conflict_description_vs_data_warns(houses):
    a = run("Predict whether the price is high - a yes or no category", houses)
    assert a.task_type == "regression"
    assert any("followed the data" in w for w in a.warnings) and a.confidence != "high"


def test_conflict_number_wording_vs_categorical_target(churn):
    a = run("estimate the churn amount, a continuous number, how much", churn)
    assert a.task_type == "classification" and any("categories" in w for w in a.warnings)


def test_garbled_and_foreign_text(churn):
    for text in ["Â¿Â¿?? Ã±andÃº æ•°æ® Ð¿Ñ€ÐµÐ´ÑÐºÐ°Ð·Ð°Ñ‚ÑŒ xqzv ###", "\x00\x01 â˜ƒ " * 50, "a" * 6000, "!!!"]:
        a = run(text, churn)
        assert a.target == "churn" and a.task_type == "classification"
    a = run("æ•¸æ“šåˆ†æž Ð¿Ñ€ÐµÐ´ÑÐºÐ°Ð·Ð°Ñ‚ÑŒ Ñ†ÐµÐ½Ñƒ", churn)
    assert any("could not make much sense" in e for e in a.explanations) and a.confidence == "medium"


def test_hints_override(houses):
    a = run("whatever", houses, hints={"target": "rooms", "task_type": "classification", "budget_minutes": 3, "priority": "small model"})
    assert a.target == "rooms" and a.task_type == "classification"
    assert a.budget_mode == "express" and a.optimisation_profile == "lightweight"


def test_bad_hint_target_warns(houses):
    a = run("predict the price", houses, hints={"target": "nonexistent"})
    assert a.target == "price" and any("nonexistent" in w for w in a.warnings)


def test_bad_task_hint_for_categorical_ignored(churn):
    a = run("", churn, hints={"task_type": "regression"})
    assert a.task_type == "classification" and any("cannot do regression" in w for w in a.warnings)


def test_unsupported_data_kind_hint(churn):
    assert any("audio" in w for w in run("", churn, hints={"data_kind": "audio"}).warnings)


@pytest.mark.parametrize("text,profile_name,mode", [
    ("this needs to run on a phone", "lightweight", "balanced"),
    ("I need to explain the results to a regulator", "interpretable", "balanced"),
    ("I want the best possible model, take your time", "accuracy_first", "deep"),
    ("quick try please", "speed_first", "express"),
    ("I have 90 minutes", "balanced", "deep"),
    ("I do not need to explain anything", "balanced", "balanced"),
])
def test_budget_and_profile_rules(churn, text, profile_name, mode):
    a = run(text, churn)
    assert (a.optimisation_profile, a.budget_mode) == (profile_name, mode)


def test_large_dataset_express():
    p = {"n_rows": 1_000_000, "n_cols": 3, "n_duplicate_rows": 0, "target_suggestions": [], "columns": [
        {"name": "x", "kind": "numeric", "dtype": "float64", "n_missing": 0, "pct_missing": 0, "n_unique": 900000, "example_values": ["0.5"], "notes": []},
        {"name": "z", "kind": "numeric", "dtype": "float64", "n_missing": 0, "pct_missing": 0, "n_unique": 900000, "example_values": ["0.5"], "notes": []},
        {"name": "target", "kind": "categorical", "dtype": "object", "n_missing": 0, "pct_missing": 0, "n_unique": 2, "example_values": ["a", "b"], "notes": []}]}
    a = advise("", p)   # profile only, no DataFrame
    assert a.target == "target" and a.budget_mode == "express" and a.metric == "macro_f1"


def test_leakage_flagged_not_excluded(churn):
    churn["churn_reason"] = np.where(churn["churn"] == "yes", "price", "none")
    churn["cancellation_date_flag"] = (churn["churn"] == "yes").astype(int)
    a = run("predict churn", churn)
    leaks = " ".join(w for w in a.warnings if "leak" in w)
    assert "churn_reason" in leaks and "cancellation_date_flag" in leaks
    assert "churn_reason" not in a.exclude_columns


def test_statistical_leak_detected(houses):
    houses["price_eur"] = houses["price"] * 0.9
    a = run("predict price", houses)
    assert any("price_eur" in w for w in a.warnings)


def test_excludes_obvious_columns():
    n = 120
    df = pd.DataFrame({"row_id": np.arange(n), "const": 1, "notes": [f"free text number {i} " + "word " * 12 for i in range(n)],
                       "x": RNG.normal(size=n), "y": RNG.normal(size=n), "target": RNG.choice(["a", "b"], n)})
    a = run("", df)
    assert set(a.exclude_columns) >= {"row_id", "const", "notes"} and "x" not in a.exclude_columns


def test_tiny_dataset_warns():
    df = pd.DataFrame({"x": RNG.normal(size=20), "target": ["a", "b"] * 10})
    assert any("rows" in w for w in run("", df).warnings)


def test_rare_class_falls_back_to_random():
    df = pd.DataFrame({"x": RNG.normal(size=100), "target": ["a"] * 99 + ["b"]})
    a = run("", df)
    assert a.split_strategy == "random" and any("rarest" in w for w in a.warnings)


def test_serialisable_roundtrip_and_determinism(churn):
    a, b = run("predict churn", churn), run("predict churn", churn)
    assert a.to_dict() == b.to_dict()
    d = json.loads(json.dumps(a.to_dict()))
    assert set(d) == {"task_type", "target", "target_candidates", "data_kind", "metric", "split_strategy", "group_column", "time_column",
                      "budget_mode", "optimisation_profile", "exclude_columns", "warnings", "explanations", "questions", "confidence"}
    assert Advice.from_dict(d).to_dict() == d
    with pytest.raises(ValueError):
        apply_answers(Advice.from_dict(d), {"target": "x"})
    assert apply_answers(Advice.from_dict(d), {"target": "churn"}, profile=profile_dataset(churn)).target == "churn"


def test_unknown_question_id_rejected(churn):
    with pytest.raises(ValueError):
        apply_answers(run("", churn), {"nope": 1})


def test_describe_advice_readable(churn):
    txt = describe_advice(run("predict churn", churn))
    assert "churn" in txt and "Why:" in txt and "confidence" in txt


def test_profile_without_columns_does_not_crash():
    a = advise("predict something", {"columns": []})
    assert a.confidence == "low" and a.warnings


def test_profile_only_works_without_dataframe(churn):
    a = run("predict churn", churn, with_df=False)
    assert a.target == "churn" and a.metric == "macro_f1"

