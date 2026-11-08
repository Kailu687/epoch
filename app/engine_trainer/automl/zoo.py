"""Model zoo: every candidate family with defaults, a tunable search space and capability flags.

Search spaces are written against a tiny sampler interface (`float`, `int`, `cat`) so they run under Optuna trials or the
built-in random sampler when Optuna is not installed.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None
HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None
HAS_OPTUNA = importlib.util.find_spec("optuna") is not None


class RandomSampler:
    """Fallback sampler with the same interface as the Optuna adapter."""

    def __init__(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def float(self, name, lo, hi, log=False):
        return float(np.exp(self.rng.uniform(np.log(lo), np.log(hi)))) if log else float(self.rng.uniform(lo, hi))

    def int(self, name, lo, hi, log=False):
        return int(round(self.float(name, lo, hi, log)))

    def cat(self, name, choices):
        return choices[self.rng.randint(len(choices))]


class TrialSampler:
    """Adapter exposing an Optuna trial through the sampler interface."""

    def __init__(self, trial):
        self.t = trial

    def float(self, name, lo, hi, log=False):
        return self.t.suggest_float(name, lo, hi, log=log)

    def int(self, name, lo, hi, log=False):
        return self.t.suggest_int(name, lo, hi, log=log)

    def cat(self, name, choices):
        return self.t.suggest_categorical(name, choices)


@dataclass
class ModelDef:
    id: str
    label: str
    family: str  # baseline | linear | tree | ensemble | boosting | kernel | neighbors | neural | bayes
    tasks: tuple
    build: Callable  # (params, task, seed, n_jobs) -> estimator
    space: Callable | None  # (sampler, task, n_rows) -> params
    defaults: dict = field(default_factory=dict)
    interpretable: bool = False
    lightweight: bool = False  # small artifact, cheap inference
    slow: bool = False
    max_rows: int | None = None  # skip above this many training rows
    description: str = ""


def _cw(params):
    return None if params.get("class_weight") in (None, "none") else params["class_weight"]


def _build_logreg(p, task, seed, n_jobs):
    return LogisticRegression(C=p.get("C", 1.0), class_weight=_cw(p), max_iter=2000)


def _build_ridge(p, task, seed, n_jobs):
    return Ridge(alpha=p.get("alpha", 1.0))


def _build_tree(p, task, seed, n_jobs):
    if task == "classification":
        return DecisionTreeClassifier(max_depth=p.get("max_depth", 6), min_samples_leaf=p.get("min_samples_leaf", 5),
                                      class_weight=_cw(p), random_state=seed)
    return DecisionTreeRegressor(max_depth=p.get("max_depth", 6), min_samples_leaf=p.get("min_samples_leaf", 5), random_state=seed)


def _build_rf(p, task, seed, n_jobs):
    kw = dict(n_estimators=p.get("n_estimators", 200), max_depth=p.get("max_depth"), min_samples_leaf=p.get("min_samples_leaf", 1),
              max_features=p.get("max_features", "sqrt" if task == "classification" else 1.0), random_state=seed, n_jobs=n_jobs)
    return RandomForestClassifier(class_weight=_cw(p), **kw) if task == "classification" else RandomForestRegressor(**kw)


def _build_et(p, task, seed, n_jobs):
    kw = dict(n_estimators=p.get("n_estimators", 300), max_depth=p.get("max_depth"), min_samples_leaf=p.get("min_samples_leaf", 1),
              max_features=p.get("max_features", "sqrt" if task == "classification" else 1.0), random_state=seed, n_jobs=n_jobs)
    return ExtraTreesClassifier(class_weight=_cw(p), **kw) if task == "classification" else ExtraTreesRegressor(**kw)


def _build_hgb(p, task, seed, n_jobs):
    kw = dict(learning_rate=p.get("learning_rate", 0.1), max_iter=p.get("max_iter", 200), max_leaf_nodes=p.get("max_leaf_nodes", 31),
              l2_regularization=p.get("l2_regularization", 0.0), min_samples_leaf=p.get("min_samples_leaf", 20), random_state=seed)
    return HistGradientBoostingClassifier(class_weight=_cw(p), **kw) if task == "classification" else HistGradientBoostingRegressor(**kw)


def _build_lgbm(p, task, seed, n_jobs):
    import lightgbm as lgb
    kw = dict(n_estimators=p.get("n_estimators", 300), learning_rate=p.get("learning_rate", 0.05), num_leaves=p.get("num_leaves", 31),
              min_child_samples=p.get("min_child_samples", 20), subsample=p.get("subsample", 0.8), subsample_freq=1,
              colsample_bytree=p.get("colsample_bytree", 0.8), reg_lambda=p.get("reg_lambda", 0.0), random_state=seed,
              n_jobs=n_jobs, verbose=-1)
    return lgb.LGBMClassifier(class_weight=_cw(p), **kw) if task == "classification" else lgb.LGBMRegressor(**kw)


def _build_xgb(p, task, seed, n_jobs):
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=p.get("n_estimators", 300), learning_rate=p.get("learning_rate", 0.05), max_depth=p.get("max_depth", 6),
                            subsample=p.get("subsample", 0.8), colsample_bytree=p.get("colsample_bytree", 0.8),
                            reg_lambda=p.get("reg_lambda", 1.0), random_state=seed, n_jobs=n_jobs, tree_method="hist")


def _build_svm(p, task, seed, n_jobs):
    if task == "classification":
        return SVC(C=p.get("C", 1.0), gamma=p.get("gamma", "scale"), probability=True, class_weight=_cw(p), random_state=seed)
    return SVR(C=p.get("C", 1.0), gamma=p.get("gamma", "scale"))


def _build_knn(p, task, seed, n_jobs):
    kw = dict(n_neighbors=p.get("n_neighbors", 7), weights=p.get("weights", "distance"), n_jobs=n_jobs)
    return KNeighborsClassifier(**kw) if task == "classification" else KNeighborsRegressor(**kw)


def _build_mlp(p, task, seed, n_jobs):
    kw = dict(hidden_layer_sizes=(p.get("units", 64),) * p.get("layers", 2), alpha=p.get("alpha", 1e-4), learning_rate_init=p.get("lr", 1e-3),
              early_stopping=True, n_iter_no_change=10, max_iter=300, random_state=seed)
    return MLPClassifier(**kw) if task == "classification" else MLPRegressor(**kw)


def _build_nb(p, task, seed, n_jobs):
    return GaussianNB(var_smoothing=p.get("var_smoothing", 1e-9))


def _build_dummy(p, task, seed, n_jobs):
    return DummyClassifier(strategy="most_frequent") if task == "classification" else DummyRegressor(strategy="mean")


CW = ("none", "balanced")
BOTH = ("classification", "regression")


def _sp_logreg(s, t, n):
    return {"C": s.float("C", 1e-3, 1e2, log=True), "class_weight": s.cat("class_weight", list(CW))}


def _sp_ridge(s, t, n):
    return {"alpha": s.float("alpha", 1e-3, 1e3, log=True)}


def _sp_tree(s, t, n):
    return {"max_depth": s.int("max_depth", 2, 14), "min_samples_leaf": s.int("min_samples_leaf", 1, 40, log=True), "class_weight": s.cat("class_weight", list(CW))}


def _sp_rf(s, t, n):
    return {"n_estimators": s.int("n_estimators", 100, 500), "max_depth": s.cat("max_depth", [None, 6, 10, 16]),
            "min_samples_leaf": s.int("min_samples_leaf", 1, 12, log=True), "max_features": s.cat("max_features", ["sqrt", 0.5, 0.8] if t == "classification" else [0.33, 0.6, 1.0]),
            "class_weight": s.cat("class_weight", list(CW))}


def _sp_hgb(s, t, n):
    return {"learning_rate": s.float("learning_rate", 0.02, 0.3, log=True), "max_iter": s.int("max_iter", 80, 400),
            "max_leaf_nodes": s.int("max_leaf_nodes", 8, 63), "l2_regularization": s.float("l2_regularization", 1e-3, 10.0, log=True),
            "min_samples_leaf": s.int("min_samples_leaf", 5, 60, log=True), "class_weight": s.cat("class_weight", list(CW))}


def _sp_lgbm(s, t, n):
    return {"n_estimators": s.int("n_estimators", 100, 600), "learning_rate": s.float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": s.int("num_leaves", 8, 96), "min_child_samples": s.int("min_child_samples", 5, 60, log=True),
            "subsample": s.float("subsample", 0.6, 1.0), "colsample_bytree": s.float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": s.float("reg_lambda", 1e-3, 10.0, log=True), "class_weight": s.cat("class_weight", list(CW))}


def _sp_xgb(s, t, n):
    return {"n_estimators": s.int("n_estimators", 100, 600), "learning_rate": s.float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": s.int("max_depth", 3, 9), "subsample": s.float("subsample", 0.6, 1.0),
            "colsample_bytree": s.float("colsample_bytree", 0.5, 1.0), "reg_lambda": s.float("reg_lambda", 1e-2, 10.0, log=True)}


def _sp_svm(s, t, n):
    return {"C": s.float("C", 1e-2, 1e2, log=True), "gamma": s.cat("gamma", ["scale", "auto"]), "class_weight": s.cat("class_weight", list(CW))}


def _sp_knn(s, t, n):
    return {"n_neighbors": s.int("n_neighbors", 3, 25), "weights": s.cat("weights", ["uniform", "distance"])}


def _sp_mlp(s, t, n):
    return {"units": s.cat("units", [32, 64, 128]), "layers": s.int("layers", 1, 3), "alpha": s.float("alpha", 1e-6, 1e-1, log=True),
            "lr": s.float("lr", 1e-4, 1e-2, log=True)}


def _sp_nb(s, t, n):
    return {"var_smoothing": s.float("var_smoothing", 1e-11, 1e-5, log=True)}


_ZOO_LIST = [
    ModelDef("baseline", "Baseline (majority / mean)", "baseline", BOTH, _build_dummy, None, interpretable=True, lightweight=True,
             description="Predicts the most common class or the average. Any useful model must beat this."),
    ModelDef("linear", "Linear model", "linear", BOTH, lambda p, t, s, n: _build_logreg(p, t, s, n) if t == "classification" else _build_ridge(p, t, s, n),
             lambda s, t, n: _sp_logreg(s, t, n) if t == "classification" else _sp_ridge(s, t, n), interpretable=True, lightweight=True,
             description="Logistic or ridge regression: simple, fast, easy to explain."),
    ModelDef("tree", "Decision tree", "tree", BOTH, _build_tree, _sp_tree, interpretable=True, lightweight=True,
             description="A single small tree: easy to read as rules."),
    ModelDef("rf", "Random forest", "ensemble", BOTH, _build_rf, _sp_rf, description="Many decision trees voting; strong general-purpose model."),
    ModelDef("extra_trees", "Extra trees", "ensemble", BOTH, _build_et, _sp_rf, description="Like a random forest with extra randomness; fast and robust."),
    ModelDef("hgb", "Gradient boosting (scikit-learn)", "boosting", BOTH, _build_hgb, _sp_hgb, description="Boosted trees built into scikit-learn."),
    ModelDef("lightgbm", "LightGBM", "boosting", BOTH, _build_lgbm, _sp_lgbm, description="Fast, accurate boosted trees; usually among the best on tables."),
    ModelDef("xgboost", "XGBoost", "boosting", ("regression",), _build_xgb, _sp_xgb, description="Boosted trees (regression only in this release)."),
    ModelDef("svm", "Support vector machine", "kernel", BOTH, _build_svm, _sp_svm, slow=True, max_rows=8000,
             description="Good on small, clean datasets; slow on large ones."),
    ModelDef("knn", "Nearest neighbours", "neighbors", BOTH, _build_knn, _sp_knn, max_rows=40000, description="Predicts from the most similar known rows."),
    ModelDef("mlp", "Neural network (MLP)", "neural", BOTH, _build_mlp, _sp_mlp, slow=True, description="A small neural network for non-linear patterns."),
    ModelDef("nb", "Naive Bayes", "bayes", ("classification",), _build_nb, _sp_nb, interpretable=True, lightweight=True,
             description="Very fast probabilistic baseline."),
]
_ORDER = ["baseline", "linear", "tree", "nb", "lightgbm", "rf", "extra_trees", "hgb", "xgboost", "knn", "svm", "mlp"]
ZOO: dict[str, ModelDef] = {m.id: m for m in sorted(_ZOO_LIST, key=lambda m: _ORDER.index(m.id))}


def available(task: str, n_rows: int, profile: str = "balanced", include_baseline: bool = True) -> list[ModelDef]:
    """Candidate families for this task and size, filtered by installed libraries and the optimisation profile."""
    out = []
    for m in ZOO.values():
        if task not in m.tasks or (m.id == "baseline" and not include_baseline):
            continue
        if m.id == "lightgbm" and not HAS_LIGHTGBM:
            continue
        if m.id == "xgboost" and not HAS_XGBOOST:
            continue
        if m.max_rows and n_rows > m.max_rows:
            continue
        if profile == "interpretable" and not m.interpretable:
            continue
        if profile == "lightweight" and not (m.lightweight or m.id in ("hgb", "lightgbm")):
            continue
        if profile == "speed_first" and m.slow:
            continue
        out.append(m)
    return out


def build(model_id: str, params: dict, task: str, seed: int = 42, n_jobs: int = 1):
    return ZOO[model_id].build(params or {}, task, seed, n_jobs)
