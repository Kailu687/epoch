"""Dataset profiling and task inference for tabular data."""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field, asdict

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype, is_string_dtype

TARGET_NAMES = re.compile(r"^(target|label|class|y|outcome|response|churn\w*|survived|species|price|diagnosis)$", re.I)
ID_NAMES = re.compile(r"(^id$|_id$|^id_|uuid|guid|^index$|^unnamed)", re.I)
MAX_ONEHOT = 50


@dataclass
class ColumnProfile:
    name: str
    kind: str  # numeric | categorical | boolean | datetime | text | constant | id
    dtype: str
    n_missing: int
    pct_missing: float
    n_unique: int
    example_values: list = field(default_factory=list)
    notes: list = field(default_factory=list)


@dataclass
class TaskSuggestion:
    target: str
    task_type: str  # classification | regression
    confidence: str  # high | medium | low
    reasons: list
    needs_confirmation: bool


def _looks_datetime(s: pd.Series) -> bool:
    if is_datetime64_any_dtype(s):
        return True
    if s.dtype != object and not is_string_dtype(s):  # pandas 3 uses a dedicated string dtype
        return False
    sample = s.dropna().astype(str).head(200)
    if len(sample) < 5 or sample.str.len().mean() > 40 or not sample.str.contains(r"\d").all():
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.95 and bool(sample.str.contains(r"[-/:]").mean() > 0.9)


def classify_column(s: pd.Series, n_rows: int) -> tuple[str, list]:
    notes = []
    nun = s.nunique(dropna=True)
    if nun <= 1:
        return "constant", ["only one distinct value"]
    if is_bool_dtype(s):
        return "boolean", notes
    if is_numeric_dtype(s):
        if ID_NAMES.search(s.name or "") and nun == s.notna().sum():
            return "id", ["unique per row and named like an identifier"]
        return "numeric", notes
    if _looks_datetime(s):
        return "datetime", ["datetime columns are not used as features in this release"]
    nonnull = s.notna().sum()
    if nonnull and nun == nonnull and nun > 20 and not _looks_datetime(s):
        return "id", ["unique per row"]
    avg_len = s.dropna().astype(str).str.len().mean() if nonnull else 0
    if nun > MAX_ONEHOT and avg_len > 30:
        return "text", ["free text is not supported as a feature in this release"]
    return "categorical", notes


def profile_columns(df: pd.DataFrame) -> list[ColumnProfile]:
    out = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        kind, notes = classify_column(s, n)
        miss = int(s.isna().sum())
        out.append(ColumnProfile(
            name=c, kind=kind, dtype=str(s.dtype), n_missing=miss, pct_missing=round(100 * miss / n, 2),
            n_unique=int(s.nunique(dropna=True)), example_values=[str(v) for v in s.dropna().unique()[:4]], notes=notes))
    return out


def suggest_targets(df: pd.DataFrame, profiles: list[ColumnProfile]) -> list[tuple[str, int, list]]:
    """Ranked (column, score, reasons). Heuristic only; the user confirms."""
    ranked = []
    last = df.columns[-1]
    for p in profiles:
        if p.kind in ("id", "text", "datetime", "constant"):
            continue
        score, why = 0, []
        if TARGET_NAMES.match(p.name):
            score += 3; why.append("name looks like a target")
        if p.name == last:
            score += 2; why.append("last column")
        if p.kind in ("categorical", "boolean") and p.n_unique <= 20:
            score += 1; why.append(f"{p.n_unique} distinct values")
        if p.pct_missing > 20:
            score -= 2; why.append("many missing values")
        if score > 0:
            ranked.append((p.name, score, why))
    return sorted(ranked, key=lambda t: -t[1])


def infer_task(df: pd.DataFrame, target: str) -> TaskSuggestion:
    s = df[target].dropna()
    nun = s.nunique()
    n = len(s)
    reasons = []
    if not is_numeric_dtype(s) or is_bool_dtype(s):
        reasons.append(f"target is non-numeric with {nun} classes")
        return TaskSuggestion(target, "classification", "high" if nun <= 50 else "medium", reasons, nun > 50)
    is_int = bool((s == s.round()).all())
    if not is_int:
        reasons.append("target has non-integer numeric values")
        return TaskSuggestion(target, "regression", "high", reasons, False)
    if nun <= 10:
        reasons.append(f"integer target with only {nun} distinct values; treated as class labels")
        return TaskSuggestion(target, "classification", "medium", reasons + ["confirm: could also be a count"], True)
    if nun / n > 0.2 or nun > 50:
        reasons.append(f"integer target with {nun} distinct values ({100 * nun / n:.0f}% of rows)")
        return TaskSuggestion(target, "regression", "medium", reasons, nun <= 20)
    reasons.append(f"integer target with {nun} distinct values; ambiguous")
    return TaskSuggestion(target, "classification", "low", reasons, True)


def class_balance(s: pd.Series) -> dict:
    vc = s.value_counts(dropna=True)
    return {"counts": {str(k): int(v) for k, v in vc.items()},
            "imbalance_ratio": round(float(vc.max() / vc.min()), 2) if len(vc) else None}


def profile_dataset(df: pd.DataFrame) -> dict:
    cols = profile_columns(df)
    return {
        "n_rows": int(len(df)), "n_cols": int(df.shape[1]),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "columns": [asdict(c) for c in cols],
        "target_suggestions": [{"column": c, "score": s, "reasons": r} for c, s, r in suggest_targets(df, cols)[:5]],
    }
