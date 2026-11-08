"""Preprocessing pipelines. Everything is fit inside sklearn Pipelines on training rows only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MAX_CATEGORIES = 30


def prepare_frame(df: pd.DataFrame, kinds: dict) -> pd.DataFrame:
    """Select feature columns and normalise dtypes. Stateless, so safe to apply before splitting.

    Numeric -> float64 (unparseable values become NaN); categorical/boolean -> str with NaN preserved.
    The same rule is applied by the exported predict example.
    """
    out = pd.DataFrame(index=df.index)
    for c, k in kinds.items():
        s = df[c]
        if k == "numeric":
            out[c] = pd.to_numeric(s, errors="coerce").astype("float64")
        else:
            out[c] = s.astype("object").where(s.isna(), s.astype(str))
    return out


def build_preprocessor(kinds: dict) -> ColumnTransformer:
    num = [c for c, k in kinds.items() if k == "numeric"]
    cat = [c for c, k in kinds.items() if k != "numeric"]
    parts = []
    if num:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=MAX_CATEGORIES, sparse_output=False)),
        ]), cat))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


def full_pipeline(kinds: dict, estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(kinds)), ("model", estimator)])
