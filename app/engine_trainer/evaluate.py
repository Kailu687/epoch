"""Metric computation. Pure functions of (model, data); the test-set policy lives in workspace.py."""
from __future__ import annotations

import numpy as np
from sklearn import metrics as M


def classification_report(model, X, y, classes) -> dict:
    pred = model.predict(X)
    classes = list(classes)
    p, r, f, s = M.precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    rep = {
        "n": int(len(y)),
        "accuracy": float(M.accuracy_score(y, pred)),
        "balanced_accuracy": float(M.balanced_accuracy_score(y, pred)),
        "macro_f1": float(M.f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(M.f1_score(y, pred, labels=classes, average="weighted", zero_division=0)),
        "per_class": {str(c): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(classes)},
        "confusion_matrix": {"labels": [str(c) for c in classes], "matrix": M.confusion_matrix(y, pred, labels=classes).tolist()},
    }
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if len(classes) == 2:
                rep["roc_auc"] = float(M.roc_auc_score(y == classes[1], proba[:, 1]))
            elif len(np.unique(y)) == len(classes):
                rep["roc_auc_ovr_macro"] = float(M.roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes))
    except Exception:
        pass
    return rep


def regression_report(model, X, y) -> dict:
    pred = model.predict(X)
    res = np.asarray(y) - pred
    return {
        "n": int(len(y)),
        "mae": float(M.mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(M.mean_squared_error(y, pred))),
        "r2": float(M.r2_score(y, pred)),
        "median_ae": float(M.median_absolute_error(y, pred)),
        "residuals": {"mean": float(res.mean()), "std": float(res.std()), "min": float(res.min()), "max": float(res.max())},
    }


def evaluate(model, X, y, task_type: str, classes=None) -> dict:
    return classification_report(model, X, y, classes) if task_type == "classification" else regression_report(model, X, y)


def primary_metric(task_type: str) -> tuple[str, bool]:
    """(name, higher_is_better)"""
    return ("macro_f1", True) if task_type == "classification" else ("rmse", False)
