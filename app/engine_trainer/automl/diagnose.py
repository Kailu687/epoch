"""Diagnosis and recommendations from measured evidence only (CV scores, learning curve, per-class results, data profile).

Every recommendation cites the numbers it came from. Nothing here claims to know why a score is low; it says what the
evidence suggests and what to try next.
"""
from __future__ import annotations

import numpy as np

from . import search as sr


def learning_curve(model_id, params, X, y, cfg, clock, fractions=(0.25, 0.5, 0.75, 1.0), splits=3):
    """CV score of the chosen model on growing fractions of the development data. Returns [(n_rows, mean, std)]."""
    out = []
    rng = np.random.RandomState(cfg.seed)
    for f in fractions:
        if clock.left() <= 0:
            break
        n = max(30, int(len(y) * f))
        idx = sr._subsample(y, cfg.task, n, cfg.seed) if f < 1 else np.arange(len(y))
        Xs, ys = X.iloc[idx].reset_index(drop=True), y[idx]
        try:
            folds = sr.make_folds(ys, cfg.task, "kfold" if cfg.cv_scheme == "group" else cfg.cv_scheme, splits, cfg.seed)
            r = sr.cv_evaluate(model_id, params, Xs, ys, folds, cfg, clock)
            out.append((int(len(idx)), r.cv_mean, r.cv_std))
        except Exception:
            break
    return out


def _rec(id_, severity, title, detail, actions, evidence):
    return {"id": id_, "severity": severity, "title": title, "detail": detail, "actions": actions, "evidence": evidence}


def recommend(*, task, metric, best, baseline_mean, curve, train_score, per_class, class_counts, profile, findings, n_dev, cv_scheme) -> list[dict]:
    recs = []
    higher = metric not in ("rmse", "mae", "log_loss")
    # 1. does the model beat doing nothing?
    floor = 0.02 * abs(baseline_mean) if (baseline_mean is not None and metric in ("rmse", "mae", "log_loss")) else 0.02
    if baseline_mean is not None and best["cv_mean"] - baseline_mean < max(floor, best["cv_std"]):
        recs.append(_rec("weak_signal", "high", "The best model barely beats the baseline",
                         "The score is close to what a do-nothing model gets, so the current columns may not contain enough information about the target.",
                         ["Check that the target column is the right one", "Add columns that describe what drives the target", "Check labels for mistakes"],
                         {"best_cv": best["cv_mean"], "baseline_cv": baseline_mean}))
    # 2. overfitting: large train vs CV gap
    if train_score is not None:
        gap = train_score - best["cv_mean"]
        if gap > max(0.15, 2 * best["cv_std"]):
            recs.append(_rec("overfitting", "high", "The model memorises the training data",
                             "It scores much higher on data it trained on than on held-out folds, so it does not generalise as well as the training score suggests.",
                             ["Add more rows", "Use a simpler or more regularised model (Epoch's Interpretable/Lightweight profile)", "Remove noisy columns"],
                             {"train_score": train_score, "cv_score": best["cv_mean"], "gap": gap}))
    # 3. learning curve
    if len(curve) >= 3:
        (n_a, s_a, _), (n_b, s_b, sd_b) = curve[-2], curve[-1]
        gain = s_b - s_a
        gain_adj = gain if higher else -gain
        if gain_adj > max(0.01, sd_b):
            recs.append(_rec("more_rows_help", "info", "More rows of the same kind will probably help",
                             f"Score rose when the training size grew from {n_a} to {n_b} rows, and the curve had not flattened.",
                             ["Collect or add more labelled rows from the same source", "Use the 'Improve this model' option to add a new dataset"],
                             {"curve": [(n, round(s, 4)) for n, s, _ in curve]}))
        elif abs(gain) <= max(0.005, 0.5 * sd_b):
            recs.append(_rec("plateau", "info", "More rows of the same kind are unlikely to help much",
                             "The score stopped improving as the training size grew, so the limit is more likely in the information the columns carry than in the amount of data.",
                             ["Add new informative columns (features)", "Add data covering situations the current rows miss", "Review label quality"],
                             {"curve": [(n, round(s, 4)) for n, s, _ in curve]}))
    # 4. weak classes
    if task == "classification" and per_class:
        weak = sorted([(c, v) for c, v in per_class.items() if v["recall"] < 0.5 and v["support"] >= 3], key=lambda t: t[1]["recall"])[:4]
        for c, v in weak:
            cnt = class_counts.get(str(c))
            recs.append(_rec(f"weak_class_{c}", "medium", f"Class '{c}' is often missed",
                             f"Only {v['recall']:.0%} of '{c}' examples were recognised in cross-validation" + (f" ({cnt} training rows)." if cnt else "."),
                             [f"Add more examples of '{c}'", f"Check whether '{c}' is confused with a similar class in the confusion matrix",
                              "Review whether the labels for this class are consistent"], {"class": str(c), "recall": v["recall"], "precision": v["precision"], "support": v["support"]}))
    # 5. instability
    if best["cv_std"] > 0.08 or (best["cv_mean"] and abs(best["cv_std"] / (abs(best["cv_mean"]) + 1e-9)) > 0.2):
        recs.append(_rec("unstable", "medium", "Results vary a lot between folds",
                         "Different parts of the data give noticeably different scores, which usually means the dataset is small or the groups differ.",
                         ["Add more rows", "Use the repeated-run option to confirm the result", "Check whether the data comes from a few sources (use a group split)"],
                         {"cv_std": best["cv_std"], "folds": best.get("fold_scores")}))
    # 6. small data
    if n_dev < 300:
        recs.append(_rec("small_data", "medium", "The dataset is small",
                         f"Only {n_dev} rows were available for training and validation, so estimates are uncertain.",
                         ["Add more rows", "Treat the reported score as a rough estimate"], {"n_dev": n_dev}))
    # 7. data quality from the profile
    for c in (profile or {}).get("columns", []):
        if c.get("pct_missing", 0) > 30 and c.get("kind") in ("numeric", "categorical", "boolean"):
            recs.append(_rec(f"missing_{c['name']}", "low", f"Column '{c['name']}' is {c['pct_missing']}% missing",
                             "Heavily missing columns carry little information and can hide patterns.", ["Improve how this column is collected", "Or drop it"],
                             {"column": c["name"], "pct_missing": c["pct_missing"]}))
    for f in findings or []:
        if f.get("code") == "possible_leakage":
            recs.append(_rec(f"leak_{f.get('column')}", "high", f"Column '{f.get('column')}' may leak the answer",
                             f["message"], ["Check whether this column is known at prediction time", "Re-run without it if not"], {"column": f.get("column")}))
        if f.get("code") == "duplicate_rows":
            recs.append(_rec("duplicates", "low", "Duplicate rows were found", f["message"], ["Remove duplicate rows before training"], {}))
    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    return sorted(recs, key=lambda r: order.get(r["severity"], 9))
