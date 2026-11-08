"""Can this new dataset be added to an existing model, and how different is it from the old data?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ..profile import classify_column


def _kind_family(kind: str) -> str:
    return "numeric" if kind == "numeric" else "categorical" if kind in ("categorical", "boolean") else kind


def check_compatibility(old_df: pd.DataFrame, new_df: pd.DataFrame, feature_kinds: dict, target: str, task: str, classes: list | None) -> dict:
    errors, warnings, drift = [], [], []
    needed = list(feature_kinds) + [target]
    missing = [c for c in needed if c not in new_df.columns]
    if missing:
        errors.append(f"The new dataset is missing column(s) the model needs: {missing}. Add them or rename the columns to match.")
        return {"ok": False, "errors": errors, "warnings": warnings, "drift": drift, "extra_columns": [], "summary": errors[0]}
    extra = [c for c in new_df.columns if c not in old_df.columns]
    if extra:
        warnings.append(f"The new dataset has extra columns that will be ignored: {extra[:8]}.")
    if len(new_df) < 10:
        errors.append("The new dataset has fewer than 10 rows.")
    for c, expected in feature_kinds.items():
        got = _kind_family(classify_column(new_df[c], len(new_df))[0])
        if got != _kind_family(expected) and got not in ("constant",):
            warnings.append(f"Column '{c}' looks {got} in the new data but was {expected} in the original. Check units and formatting.")
        if new_df[c].isna().mean() > 0.5 and old_df[c].isna().mean() <= 0.5:
            warnings.append(f"Column '{c}' is mostly empty in the new data ({new_df[c].isna().mean():.0%}).")
        # drift
        a, b = old_df[c].dropna(), new_df[c].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        if expected == "numeric":
            a, b = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            ks = float(ks_2samp(a.sample(min(len(a), 5000), random_state=0), b.sample(min(len(b), 5000), random_state=0)).statistic)
            shift = float(abs(b.mean() - a.mean()) / (a.std() + 1e-9))
            drift.append({"column": c, "kind": "numeric", "score": ks, "note": f"distribution distance {ks:.2f}; mean moved {shift:.2f} standard deviations"})
        else:
            pa, pb = a.astype(str).value_counts(normalize=True), b.astype(str).value_counts(normalize=True)
            idx = pa.index.union(pb.index)
            tv = float(0.5 * np.abs(pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).sum())
            unseen = float(pb[[k for k in pb.index if k not in pa.index]].sum()) if len(pb) else 0.0
            drift.append({"column": c, "kind": "categorical", "score": tv,
                          "note": f"category mix differs by {tv:.2f}" + (f"; {unseen:.0%} of rows use categories never seen before" if unseen > 0 else "")})
    drift.sort(key=lambda d: -d["score"])
    if task == "classification":
        new_labels = set(new_df[target].dropna().unique().tolist())
        unseen = sorted(map(str, new_labels - set(classes or [])))
        if unseen:
            warnings.append(f"The new data contains classes the model has not seen: {unseen[:6]}. Epoch will retrain with them included (the label set changes).")
        pa = old_df[target].astype(str).value_counts(normalize=True)
        pb = new_df[target].astype(str).value_counts(normalize=True)
        idx = pa.index.union(pb.index)
        tv = float(0.5 * np.abs(pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).sum())
        if tv > 0.15:
            warnings.append(f"The class mix is quite different in the new data (difference {tv:.2f}).")
    else:
        a, b = pd.to_numeric(old_df[target], errors="coerce").dropna(), pd.to_numeric(new_df[target], errors="coerce").dropna()
        if len(a) > 5 and len(b) > 5 and abs(b.mean() - a.mean()) > 1.0 * a.std():
            warnings.append("The target values in the new data are centred noticeably differently from the original data.")
    strong = [d for d in drift if d["score"] > 0.3]
    if strong:
        warnings.append(f"{len(strong)} column(s) look quite different in the new data (e.g. '{strong[0]['column']}': {strong[0]['note']}). "
                        "The new data may come from a different source; check the comparison carefully.")
    summary = ("Compatible." if not errors else errors[0]) + (f" {len(warnings)} warning(s)." if warnings else "")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "drift": drift[:12], "extra_columns": extra, "summary": summary}
