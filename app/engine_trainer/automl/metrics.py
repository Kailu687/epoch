"""Selection metrics. `score()` always returns higher-is-better so search code never has to branch on direction."""
from __future__ import annotations

import numpy as np
from sklearn import metrics as M

CLASSIFICATION = ("macro_f1", "balanced_accuracy", "accuracy", "roc_auc", "pr_auc", "log_loss")
REGRESSION = ("rmse", "mae", "r2")
NEEDS_PROBA = {"roc_auc", "pr_auc", "log_loss"}
LOWER_IS_BETTER = {"rmse", "mae", "log_loss"}


def default_metric(task: str, imbalance: float | None = None) -> str:
    if task == "regression":
        return "rmse"
    return "macro_f1" if (imbalance or 1) > 1.5 else "balanced_accuracy" if (imbalance or 1) > 1.2 else "accuracy"


def valid_metric(task: str, metric: str | None) -> str:
    allowed = CLASSIFICATION if task == "classification" else REGRESSION
    return metric if metric in allowed else default_metric(task)


def raw(metric: str, y, pred, proba=None, classes=None) -> float:
    """Metric in its natural direction (use `score` for selection)."""
    if metric == "macro_f1":
        return float(M.f1_score(y, pred, average="macro", zero_division=0))
    if metric == "balanced_accuracy":
        return float(M.balanced_accuracy_score(y, pred))
    if metric == "accuracy":
        return float(M.accuracy_score(y, pred))
    if metric == "rmse":
        return float(np.sqrt(M.mean_squared_error(y, pred)))
    if metric == "mae":
        return float(M.mean_absolute_error(y, pred))
    if metric == "r2":
        return float(M.r2_score(y, pred))
    if proba is None:
        raise ValueError(f"metric {metric} needs class probabilities")
    classes = list(classes) if classes is not None else sorted(set(y))
    if metric == "log_loss":
        return float(M.log_loss(y, proba, labels=classes))
    if len(classes) == 2:
        pos = np.asarray(y) == classes[1]
        return float(M.roc_auc_score(pos, proba[:, 1]) if metric == "roc_auc" else M.average_precision_score(pos, proba[:, 1]))
    if metric == "roc_auc":
        return float(M.roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes))
    # multiclass PR-AUC: macro one-vs-rest average precision
    y = np.asarray(y)
    return float(np.mean([M.average_precision_score(y == c, proba[:, i]) for i, c in enumerate(classes)]))


def score(metric: str, y, pred, proba=None, classes=None) -> float:
    v = raw(metric, y, pred, proba, classes)
    return -v if metric in LOWER_IS_BETTER else v


def report_from_predictions(task: str, y, pred, proba=None, classes=None) -> dict:
    """Full report in the same shape as `evaluate.evaluate()` but from stored predictions (used for out-of-fold results)."""
    y = np.asarray(y)
    if task == "regression":
        res = y - pred
        return {"n": int(len(y)), "mae": raw("mae", y, pred), "rmse": raw("rmse", y, pred), "r2": raw("r2", y, pred),
                "median_ae": float(M.median_absolute_error(y, pred)),
                "residuals": {"mean": float(res.mean()), "std": float(res.std()), "min": float(res.min()), "max": float(res.max())}}
    classes = list(classes)
    p, r, f, s = M.precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    rep = {"n": int(len(y)), "accuracy": raw("accuracy", y, pred), "balanced_accuracy": raw("balanced_accuracy", y, pred),
           "macro_f1": raw("macro_f1", y, pred),
           "weighted_f1": float(M.f1_score(y, pred, labels=classes, average="weighted", zero_division=0)),
           "per_class": {str(c): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(classes)},
           "confusion_matrix": {"labels": [str(c) for c in classes], "matrix": M.confusion_matrix(y, pred, labels=classes).tolist()}}
    if proba is not None:
        for m in ("roc_auc", "pr_auc", "log_loss"):
            try:
                rep[m] = raw(m, y, pred, proba, classes)
            except Exception:
                pass
    return rep
