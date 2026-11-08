"""Model providers. A provider proposes candidate specs and builds estimators for a task."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge


@dataclass
class CandidateSpec:
    id: str
    label: str
    estimator: str
    params: dict = field(default_factory=dict)
    is_baseline: bool = False
    description: str = ""


class ModelProvider(Protocol):
    name: str

    def supports(self, task_type: str, domain: str) -> bool: ...
    def candidates(self, task_type: str, n_rows: int, imbalance: float | None, seed: int, n_jobs: int) -> list[CandidateSpec]: ...
    def build(self, spec: CandidateSpec): ...


class SklearnTabularProvider:
    name = "sklearn-tabular"

    def supports(self, task_type, domain):
        return domain == "tabular" and task_type in ("classification", "regression")

    def candidates(self, task_type, n_rows, imbalance, seed, n_jobs):
        c: list[CandidateSpec] = []
        if task_type == "classification":
            cw = "balanced" if imbalance and imbalance > 3 else None
            c.append(CandidateSpec("baseline", "Baseline (most frequent class)", "DummyClassifier", {"strategy": "most_frequent"}, True,
                                   "Predicts the majority class. Any useful model must beat this."))
            c.append(CandidateSpec("logreg", "Logistic regression", "LogisticRegression", {"max_iter": 1000, "class_weight": cw},
                                   description="Linear model on scaled/one-hot features."))
            c.append(CandidateSpec("rf", "Random forest", "RandomForestClassifier",
                                   {"n_estimators": 200, "class_weight": cw, "random_state": seed, "n_jobs": n_jobs},
                                   description="Bagged decision trees."))
            c.append(CandidateSpec("hgb", "Gradient boosting (histogram)", "HistGradientBoostingClassifier",
                                   {"random_state": seed, "class_weight": cw}, description="Boosted trees; fast on larger tables."))
        else:
            c.append(CandidateSpec("baseline", "Baseline (mean)", "DummyRegressor", {"strategy": "mean"}, True,
                                   "Predicts the training mean. Any useful model must beat this."))
            c.append(CandidateSpec("ridge", "Ridge regression", "Ridge", {"alpha": 1.0}, description="Regularised linear model."))
            c.append(CandidateSpec("rf", "Random forest", "RandomForestRegressor",
                                   {"n_estimators": 200, "random_state": seed, "n_jobs": n_jobs}, description="Bagged decision trees."))
            c.append(CandidateSpec("hgb", "Gradient boosting (histogram)", "HistGradientBoostingRegressor",
                                   {"random_state": seed}, description="Boosted trees; fast on larger tables."))
        return c

    def build(self, spec):
        cls = {"DummyClassifier": DummyClassifier, "DummyRegressor": DummyRegressor, "LogisticRegression": LogisticRegression,
               "Ridge": Ridge, "RandomForestClassifier": RandomForestClassifier, "RandomForestRegressor": RandomForestRegressor,
               "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
               "HistGradientBoostingRegressor": HistGradientBoostingRegressor}[spec.estimator]
        return cls(**spec.params)


PROVIDERS: list[ModelProvider] = [SklearnTabularProvider()]


def provider_for(task_type: str, domain: str = "tabular") -> ModelProvider:
    for p in PROVIDERS:
        if p.supports(task_type, domain):
            return p
    raise ValueError(f"No model provider supports {task_type!r} on {domain!r} data (planned, not implemented).")
