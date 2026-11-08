"""Data-driven checks used by the Advisor: groups, time, leakage, balance, outliers.

All functions are pure. Those that take ``df`` also work (with fewer signals) when
``df`` is None, using only the profile dict.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .language import split_name, stem

GROUP_NAME = re.compile(r"(patient|subject|speaker|user|customer|client|machine|device|session|person|participant|household|"
                        r"student|driver|player|animal|sensor|member|account|employee|family|visitor|child)", re.I)
TIME_NAME = re.compile(r"^(year|yr|date|time|timestamp|period|week|day|month|step|t|epoch|ts|datetime)$|"
                       r"_(year|date|time|timestamp|period|week|day|month|step|datetime)$", re.I)
STRONG_TARGET = re.compile(r"^(target|label|class|y|outcome|response|churn(?:ed)?|survived|species|price|diagnosis|fraud(?:ulent)?|"
                           r"default(?:ed)?|attrition|salary|revenue|medv|sale_?price|selling_?price|is_fraud|rent|cost)$", re.I)
WEAK_TARGET = re.compile(r"^(rating|grade|score|status|result|quality|risk|income|amount|sales|profit|category|level|fare|demand)$", re.I)
POST_OUTCOME = re.compile(r"(after|post_|_post|resolution|resolved|closed|refund|cancell?ation|churn_(?:reason|date)|discharge|"
                          r"settle|final|actual|follow_?up)", re.I)
SAMPLE_ROWS = 20000


def eligible_target_kinds() -> tuple:
    return ("numeric", "categorical", "boolean")


def sample_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic head-based sample so large frames stay cheap."""
    return df if len(df) <= SAMPLE_ROWS else df.iloc[:: max(1, len(df) // SAMPLE_ROWS)].head(SAMPLE_ROWS)


def is_monotonic(s: pd.Series) -> bool:
    """True if the column is non-decreasing (or non-increasing) with >1 distinct value; dates are parsed."""
    s = s.dropna()
    if len(s) < 10 or s.nunique() < 2:
        return False
    if not is_numeric_dtype(s):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce")
        if parsed.isna().mean() > 0.05:
            return False
        s = parsed.dropna()
    return bool(s.is_monotonic_increasing or s.is_monotonic_decreasing)


def time_candidates(cols: dict, df: pd.DataFrame | None, exclude: set) -> list[dict]:
    """Ranked datetime-like or monotonic time columns: [{column, score, monotonic, reasons}]."""
    out = []
    for name, p in cols.items():
        if name in exclude:
            continue
        kind, score, why = p.get("kind"), 0.0, []
        mono = False
        if kind == "datetime":
            score += 2; why.append("holds dates/times")
        elif kind == "numeric" and df is not None and TIME_NAME.search(name):
            why.append("numeric column with a time-like name")
        else:
            continue
        if TIME_NAME.search(name):
            score += 1; why.append("name looks like a time column")
        if df is not None and name in df.columns:
            mono = is_monotonic(df[name])
            if mono:
                score += 2; why.append("rows are already in time order")
            elif kind == "numeric":
                continue
        if score > 0:
            out.append({"column": name, "score": score, "monotonic": mono, "reasons": why})
    return sorted(out, key=lambda d: (-d["score"], d["column"]))


def group_candidates(cols: dict, n_rows: int, exclude: set, entities: set, mentions: dict) -> list[dict]:
    """Columns that look like a repeated entity (patient, speaker, customer, machine...)."""
    out = []
    for name, p in cols.items():
        if name in exclude or p.get("kind") not in ("categorical", "numeric", "boolean"):
            continue
        nun = int(p.get("n_unique") or 0)
        if nun < 2 or n_rows <= 0 or nun > 0.9 * n_rows or n_rows / nun < 1.5:
            continue
        words = {stem(w) for w in split_name(name)}
        by_name = bool(GROUP_NAME.search(name))
        by_text = bool(words & entities)
        if not (by_name or by_text):
            continue
        score = (2.0 if by_name else 0.0) + (3.0 if by_text else 0.0) + (1.0 if name in mentions else 0.0)
        why = [f"{nun} distinct values over {n_rows} rows (about {n_rows / nun:.1f} rows each)"]
        if by_name:
            why.append("name looks like an entity identifier")
        if by_text:
            why.append("matches the entity you described")
        out.append({"column": name, "score": score, "n_groups": nun, "reasons": why})
    return sorted(out, key=lambda d: (-d["score"], d["column"]))


def repeated_columns(cols: dict, n_rows: int, exclude: set, limit: int = 8) -> list[str]:
    """Fallback options for 'which column identifies the same person/machine?'."""
    c = [(n, int(p.get("n_unique") or 0)) for n, p in cols.items()
         if n not in exclude and p.get("kind") in ("categorical", "numeric") and 2 <= int(p.get("n_unique") or 0) <= 0.9 * max(n_rows, 1)
         and n_rows / max(int(p.get("n_unique") or 1), 1) >= 1.5]
    return [n for n, _ in sorted(c, key=lambda t: (t[1], t[0]))][:limit]


def name_target_score(name: str) -> tuple[float, str | None]:
    """Score how much a column NAME looks like an outcome (strong 3.5, weak 1.5)."""
    if STRONG_TARGET.match(name):
        return 3.5, "name looks like something you would predict"
    if WEAK_TARGET.match(name):
        return 1.5, "name could be an outcome"
    return 0.0, None


def leakage_suspects(df: pd.DataFrame | None, cols: dict, target: str, skip: set, task: str) -> list[tuple[str, str]]:
    """Columns that look like they reveal the answer. Warning-only heuristics.

    Name check: sounds post-outcome, or contains the target's name. Statistics check
    (needs df): a single column that almost perfectly predicts the target.
    """
    out: list[tuple[str, str]] = []
    tstem = [w for w in split_name(target) if len(w) >= 4]
    for name in cols:
        if name == target or name in skip:
            continue
        if POST_OUTCOME.search(name):
            out.append((name, "its name sounds like something that is only known after the outcome"))
        elif tstem and any(w in split_name(name) or any(w in x for x in split_name(name)) for w in tstem):
            out.append((name, f"its name contains '{target}', so it may be a copy or consequence of the answer"))
    flagged = {n for n, _ in out}
    if df is None or target not in df.columns:
        return out[:6]
    d = sample_frame(df)
    y = d[target]
    checked = 0
    for name, p in cols.items():
        if name == target or name in skip or name in flagged or name not in d.columns or p.get("kind") not in ("numeric", "categorical", "boolean"):
            continue
        checked += 1
        if checked > 60:
            break
        try:
            r = _single_column_power(d[name], y, task)
        except Exception:  # pragma: no cover - defensive, statistics must never crash advice
            r = None
        if r:
            out.append((name, r))
    return out[:6]


def _single_column_power(x: pd.Series, y: pd.Series, task: str) -> str | None:
    m = x.notna() & y.notna()
    if m.sum() < 40:
        return None
    x, y = x[m], y[m]
    if task == "regression":
        if not (is_numeric_dtype(x) and is_numeric_dtype(y)) or x.nunique() < 3 or y.nunique() < 3:
            return None
        rho = x.rank().corr(y.rank())
        if rho == rho and abs(rho) >= 0.995:
            return f"it moves almost exactly in step with the answer (rank correlation {rho:.3f})"
        return None
    nun = x.nunique()
    if nun > len(x) / 4 and not is_numeric_dtype(x):
        return None
    xb = x
    if is_numeric_dtype(x) and nun > 20:
        xb = pd.qcut(x.rank(method="first"), 20, labels=False, duplicates="drop")
    elif nun > len(x) / 4:
        return None
    ct = pd.crosstab(xb, y)
    purity = ct.max(axis=1).sum() / ct.values.sum()
    base = y.value_counts(normalize=True).iloc[0]
    if purity >= 0.99 and purity - base >= 0.05:
        return "on its own it predicts the answer almost perfectly, which is rare in real data"
    return None


def class_stats(y: pd.Series) -> dict:
    """Counts, number of classes, imbalance ratio and minority share for a class column."""
    vc = y.dropna().value_counts()
    if vc.empty:
        return {"n_classes": 0, "ratio": None, "minority_share": None, "min_count": 0, "counts": {}}
    return {"n_classes": int(len(vc)), "ratio": float(vc.max() / vc.min()), "minority_share": float(vc.min() / vc.sum()),
            "min_count": int(vc.min()), "counts": {str(k): int(v) for k, v in vc.items()}, "minority": str(vc.index[-1])}


def heavy_outliers(y: pd.Series) -> float:
    """Share of values beyond 3 IQR from the quartiles (a crude 'heavy tail' measure)."""
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(y) < 20:
        return 0.0
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return 0.0
    return float(((y < q1 - 3 * iqr) | (y > q3 + 3 * iqr)).mean())
