"""Dataset/task validation. Findings are errors (block training), warnings or info."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .profile import ColumnProfile, class_balance

USABLE = ("numeric", "categorical", "boolean")
LEAK_SCORE = 0.97


@dataclass
class Finding:
    severity: str  # error | warning | info
    code: str
    message: str
    column: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class ValidationResult:
    findings: list
    features: list  # usable feature column names
    feature_kinds: dict  # name -> numeric|categorical|boolean
    row_mask: np.ndarray  # rows kept (target present, optional de-dup)

    @property
    def errors(self):
        return [f for f in self.findings if f.severity == "error"]

    @property
    def ok(self):
        return not self.errors


def validate(df: pd.DataFrame, profiles: list[ColumnProfile], target: str, task_type: str,
             exclude: list[str] | None = None, include: list[str] | None = None, drop_duplicates: bool = False) -> ValidationResult:
    F: list[Finding] = []
    exclude, include = set(exclude or []), set(include or [])
    by_name = {p.name: p for p in profiles}
    if target not in by_name:
        return ValidationResult([Finding("error", "target_missing", f"Target column '{target}' not found.")], [], {}, np.zeros(len(df), bool))
    tp = by_name[target]
    y = df[target]
    mask = y.notna().to_numpy()
    if (~mask).any():
        F.append(Finding("warning", "target_missing_values", f"{int((~mask).sum())} rows have no target value and were dropped.", target))
    if tp.kind in ("id", "text", "datetime", "constant"):
        F.append(Finding("error", "bad_target", f"Target '{target}' looks like a {tp.kind} column and cannot be predicted by this release.", target))
    if task_type == "regression" and tp.kind != "numeric":
        F.append(Finding("error", "regression_nonnumeric", "Regression needs a numeric target.", target))
    if drop_duplicates:
        dup = df.duplicated().to_numpy()
        if dup.any():
            F.append(Finding("info", "duplicates_dropped", f"{int(dup.sum())} duplicate rows were removed before splitting."))
            mask &= ~dup
    else:
        nd = int(df.duplicated().sum())
        if nd:
            F.append(Finding("warning", "duplicate_rows", f"{nd} duplicate rows. Identical rows can land in both train and test sets and inflate scores; consider removing duplicates."))

    features, kinds = [], {}
    for p in profiles:
        if p.name == target or p.name in exclude:
            continue
        if p.name in include and p.kind in ("id", "constant"):
            pass
        elif p.kind in ("id", "text", "datetime", "constant"):
            F.append(Finding("warning" if p.kind == "id" else "info", f"excluded_{p.kind}", f"Column '{p.name}' excluded: {'; '.join(p.notes)}.", p.name))
            continue
        if p.pct_missing >= 100:
            F.append(Finding("warning", "all_missing", f"Column '{p.name}' is entirely missing and was excluded.", p.name))
            continue
        if p.pct_missing > 50:
            F.append(Finding("warning", "mostly_missing", f"Column '{p.name}' is {p.pct_missing}% missing.", p.name))
        kind = "numeric" if p.kind in ("numeric", "id", "constant") else p.kind
        features.append(p.name); kinds[p.name] = kind
        t = target.lower()
        if len(t) >= 3 and t in p.name.lower():
            F.append(Finding("warning", "leak_name", f"Column '{p.name}' has a name containing the target; it may be derived from it.", p.name))
    if not features:
        F.append(Finding("error", "no_features", "No usable feature columns remain."))
    n = int(mask.sum())
    if n < 30:
        F.append(Finding("error", "too_few_rows", f"Only {n} usable rows; at least 30 are needed for a train/validation/test split."))
    if task_type == "classification" and tp.kind not in ("id", "text", "datetime", "constant"):
        cb = class_balance(y[mask])
        counts = cb["counts"]
        k = len(counts)
        if k < 2:
            F.append(Finding("error", "one_class", "The target has fewer than two classes.", target))
        elif k > 100:
            F.append(Finding("error", "too_many_classes", f"{k} classes; this release supports up to 100.", target))
        rare = [c for c, v in counts.items() if v < 3]
        if rare:
            F.append(Finding("error", "rare_classes", f"Classes with fewer than 3 examples cannot be split: {rare[:5]}. Remove or merge them.", target))
        if cb["imbalance_ratio"] and cb["imbalance_ratio"] > 3:
            F.append(Finding("warning", "imbalance", f"Class imbalance ratio {cb['imbalance_ratio']}:1. Accuracy is misleading; macro-F1 and balanced accuracy are reported and used for selection.", target))
    return ValidationResult(F, features, kinds, mask)


def leakage_probe(X: pd.DataFrame, y: np.ndarray, kinds: dict, task_type: str, seed: int = 0, max_rows: int = 20000) -> list[Finding]:
    """Single-feature predictive power on TRAINING rows only. Heuristic: flags, never removes."""
    out = []
    n = len(y)
    if n > max_rows:
        sel = np.random.RandomState(seed).choice(n, max_rows, replace=False)
        X, y = X.iloc[sel], y[sel]
    strat = y if task_type == "classification" else None
    try:
        a, b = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed, stratify=strat)
    except ValueError:
        a, b = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed)
    for c in X.columns:
        col = X[[c]]
        try:
            if kinds[c] == "numeric":
                pre = SimpleImputer(strategy="median")
            else:
                pre = make_pipeline(SimpleImputer(strategy="constant", fill_value="__missing__", missing_values=np.nan),
                                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
                col = col.astype(object).where(col.notna(), np.nan)
            if task_type == "classification":
                m = make_pipeline(pre, DecisionTreeClassifier(max_depth=6, random_state=seed))
                m.fit(col.iloc[a], y[a])
                score = balanced_accuracy_score(y[b], m.predict(col.iloc[b]))
            else:
                m = make_pipeline(pre, DecisionTreeRegressor(max_depth=8, random_state=seed))
                m.fit(col.iloc[a], y[a])
                score = r2_score(y[b], m.predict(col.iloc[b]))
        except Exception:
            continue
        if score >= LEAK_SCORE:
            out.append(Finding("warning", "possible_leakage",
                               f"Column '{c}' alone predicts the target almost perfectly on held-out training rows "
                               f"({'balanced accuracy' if task_type == 'classification' else 'R²'} {score:.3f}). "
                               "This may be target leakage (information unavailable at prediction time). Review it before trusting results.", c))
    return out
