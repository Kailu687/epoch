"""AutoML experiment runner (v2). Same event protocol as `runner.run_plan`, richer search.

Flow: load -> profile/validate -> split (stratified | group | temporal) -> carve the locked test set -> search on the
development rows (train+validation) with cross-validation -> refit finalists -> choose -> diagnose -> summary.
The test rows are never passed to anything in this module.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, VotingRegressor

from .. import pipeline as pl
from ..io_csv import DatasetError, read_csv, sha256_file
from ..profile import class_balance, profile_columns
from ..splits import SplitError, make_split_strategy, split_hash
from ..validate import leakage_probe, validate
from . import diagnose, zoo
from . import metrics as mx
from . import search as sr


def _py(v):
    return v.item() if hasattr(v, "item") else v


def _order_values(series: pd.Series) -> np.ndarray:
    """Sortable time values from a datetime-like or numeric column."""
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.95:
        return num.fillna(num.median()).to_numpy()
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().mean() < 0.95:
        raise SplitError("The time column could not be read as numbers or dates.")
    return dt.fillna(dt.min()).astype("int64").to_numpy()


def choose(finals: list, ens: dict | None, profile: str) -> tuple[str, str]:
    """Return (candidate_id, reason). Prefer the best CV score; under 'balanced' prefer a cheaper model within a small margin."""
    pool = {r.model_id: (r.cv_mean, r.cv_std, r.fit_seconds) for r in finals}
    if ens and ens["accepted"]:
        pool["ensemble"] = (ens["score"], finals[0].cv_std, sum(r.fit_seconds for r in finals))
    best_id = max(pool, key=lambda k: pool[k][0])
    best = pool[best_id]
    if profile in ("balanced", "speed_first", "lightweight") and len(pool) > 1:
        tol = max(0.002, 0.25 * best[1])
        near = [k for k, v in pool.items() if v[0] >= best[0] - tol]
        cheap = min(near, key=lambda k: pool[k][2])
        if cheap != best_id:
            return cheap, (f"{zoo.ZOO[cheap].label if cheap in zoo.ZOO else cheap} scored within {tol:.3f} of the best ({zoo.ZOO[best_id].label if best_id in zoo.ZOO else 'ensemble'}) "
                           "and is cheaper to train and run, so it was chosen.")
    if best_id == "ensemble":
        return best_id, f"A weighted ensemble of {', '.join(m for m, _ in ens['members'])} beat the best single model by {ens['gain']:.3f} (more than the {ens['margin']:.3f} needed)."
    return best_id, f"{zoo.ZOO[best_id].label} had the best cross-validated score ({best[0]:.4f})."


def run_automl(plan: dict, emit: Callable[[dict], None], cancelled: Callable[[], bool] = lambda: False) -> dict:
    """Run under bounded thread pools: spawned workers cannot use loky, and unbounded OpenMP threads slow small jobs a lot."""
    import joblib
    from threadpoolctl import threadpool_limits
    n = max(1, int(plan.get("n_jobs", 1)))
    with joblib.parallel_backend("threading", n_jobs=n), threadpool_limits(limits=n):
        return _run_automl(plan, emit, cancelled)


def _run_automl(plan: dict, emit: Callable[[dict], None], cancelled: Callable[[], bool]) -> dict:
    run_dir = Path(plan["run_dir"])
    (run_dir / "candidates").mkdir(parents=True, exist_ok=True)
    ev = lambda type_, **kw: emit({"type": type_, "ts": time.time(), **kw})
    task = plan["task_type"]
    t_start = time.perf_counter()
    try:
        ev("stage", name="loading")
        df, info = read_csv(plan["dataset_path"])
        if plan.get("dataset_sha256") and info.sha256 != plan["dataset_sha256"]:
            raise DatasetError("Dataset file changed since import (checksum mismatch).")
        profiles = profile_columns(df)
        group_col, time_col = plan.get("group_column"), plan.get("time_column")
        for c in (group_col, time_col):
            if c and c not in df.columns:
                raise DatasetError(f"Column '{c}' (group/time) not found in the dataset.")
        exclude = list(plan.get("exclude") or []) + [c for c in (group_col, time_col) if c]
        ev("stage", name="validating")
        vr = validate(df, profiles, plan["target"], task, exclude, plan.get("include"), plan.get("drop_duplicates", False))
        ev("findings", findings=[f.to_dict() for f in vr.findings])
        if not vr.ok:
            ev("failed", reason="Validation errors: " + " | ".join(f.message for f in vr.errors))
            return {"status": "failed"}

        rows = np.flatnonzero(vr.row_mask)
        X = pl.prepare_frame(df.iloc[rows], vr.feature_kinds).reset_index(drop=True)
        y_raw = df[plan["target"]].iloc[rows].reset_index(drop=True)
        if task == "classification":
            classes = [_py(c) for c in sorted(y_raw.unique().tolist(), key=lambda v: (str(type(v)), v))]
            y = np.array([_py(v) for v in y_raw], dtype=object if any(isinstance(c, str) for c in classes) else None)
            counts = class_balance(y_raw)
            imbalance = counts["imbalance_ratio"]
        else:
            classes, y, imbalance, counts = None, y_raw.to_numpy(dtype="float64"), None, {"counts": {}}
        groups = df[group_col].iloc[rows].astype(str).to_numpy() if group_col else None
        order = _order_values(df[time_col].iloc[rows]) if time_col else None

        strategy = plan.get("split_strategy") or "stratified"
        ev("stage", name="splitting")
        holdout = np.array([], dtype=int)
        cf = plan.get("continue_from")
        if cf:  # improving an existing model: SAME locked test rows as before; some of the new rows are held out as a second, separate test
            from sklearn.model_selection import train_test_split
            test_pos = np.flatnonzero(np.isin(rows, cf["old_test_rows"]))
            new_pos = np.flatnonzero(rows >= cf["n_old_rows"])
            n_hold = max(5, int(round(len(new_pos) * cf.get("holdout_frac", 0.15))))
            if len(new_pos) < 20:
                raise SplitError("The new dataset needs at least 20 usable rows to be added.")
            try:
                _, holdout = train_test_split(new_pos, test_size=n_hold, random_state=plan["seed"], stratify=y[new_pos] if task == "classification" else None)
            except ValueError:
                _, holdout = train_test_split(new_pos, test_size=n_hold, random_state=plan["seed"])
            rest = np.setdiff1d(np.arange(len(rows)), np.concatenate([test_pos, holdout]))
            split = {"train": np.sort(rest), "val": np.array([], dtype=int), "test": np.sort(test_pos)}
            strategy = "continue"
        else:
            split = make_split_strategy(y, task, strategy, plan["seed"], plan.get("val_frac", 0.15), plan.get("test_frac", 0.15), groups, order)
        sh = split_hash(split)
        dev = np.sort(np.concatenate([split["train"], split["val"]]))
        if strategy == "temporal":
            dev = dev[np.argsort(order[dev], kind="stable")]
        (run_dir / "split.json").write_text(json.dumps({
            "seed": plan["seed"], "hash": sh, "strategy": strategy, "group_column": group_col, "time_column": time_col,
            "indices_refer_to": "rows kept after cleaning; original_rows maps them to CSV data rows (0-based)",
            "original_rows": rows.tolist(), "dev": dev.tolist(), "new_holdout": holdout.tolist(), **{k: v.tolist() for k, v in split.items()}}), encoding="utf-8")
        ev("split", sizes={**{k: int(len(v)) for k, v in split.items()}, "dev": int(len(dev)), **({"new_holdout": int(len(holdout))} if cf else {})}, hash=sh, strategy=strategy)

        Xd, yd = X.iloc[dev].reset_index(drop=True), y[dev]
        gd = None if groups is None else groups[dev]

        ev("stage", name="leakage-check")
        leak = leakage_probe(Xd, yd, vr.feature_kinds, task, plan["seed"])
        if leak:
            ev("findings", findings=[f.to_dict() for f in leak])
        all_findings = [f.to_dict() for f in vr.findings + leak]

        metric = mx.valid_metric(task, plan.get("metric")) if plan.get("metric") else mx.default_metric(task, imbalance)
        mode = plan.get("budget_mode", "express")
        budget = float(plan.get("budget_seconds") or sr.BUDGET_SECONDS.get(mode, 150))
        profile_name = plan.get("optimisation_profile", "balanced")
        scheme = {"group": "group", "temporal": "temporal"}.get(strategy, "stratified" if task == "classification" else "kfold")
        cfg = sr.SearchConfig(task=task, metric=metric, kinds=vr.feature_kinds, classes=classes, seed=plan["seed"], n_jobs=plan.get("n_jobs", 1),
                              budget_seconds=budget, mode=mode, profile=profile_name, cv_scheme=scheme, n_splits=plan.get("cv_folds", 5),
                              max_trials=plan.get("max_trials"), n_finalists=plan.get("n_finalists"), only=plan.get("models"))
        defs = zoo.available(task, len(dev), profile_name)
        if plan.get("models"):
            defs = [d for d in defs if d.id in plan["models"] or d.id == "baseline"]
        ev("plan", candidates=[{"id": d.id, "label": d.label, "estimator": d.family, "params": d.defaults, "is_baseline": d.id == "baseline",
                                "description": d.description} for d in defs], metric=metric, scheme=scheme, budget_seconds=budget, mode=mode)

        clock = sr.Clock(budget, cancelled)
        ev("stage", name="searching")

        def sub_emit(kind, payload):
            ev(kind, **payload)

        res = sr.run_search(Xd, yd, gd, cfg, clock, sub_emit)
        finals, ens = res["finalists"], res["ensemble"]
        if not finals:
            ev("failed", reason="No model could be trained on this data. " + " ".join(res["log"]))
            return {"status": "failed"}

        # ---- baseline reference + screened-only candidates
        base = next((r for r in res["results"] if r.model_id == "baseline"), None)
        base_mean = base.cv_mean if base else None
        final_ids = {r.model_id for r in finals}
        for r in res["results"]:
            if r.model_id in final_ids or r.model_id == "baseline" or r.status != "ok":
                continue
            ev("candidate_screened", id=r.model_id, cv_mean=r.cv_mean, cv_std=r.cv_std,
               score=(-r.cv_mean if metric in mx.LOWER_IS_BETTER else r.cv_mean), seconds=round(r.fit_seconds, 2))

        # ---- refit finalists on all development rows and save
        ev("stage", name="refitting")
        results_out, models = [], {}
        higher = metric not in mx.LOWER_IS_BETTER

        def make_report(r_or_ens, oof_pred, oof_proba, covered):
            cov = covered
            return mx.report_from_predictions(task, yd[cov], oof_pred[cov], None if oof_proba is None else oof_proba[cov], classes)

        def save(cid, model, fit_s, rep, extra):
            path = run_dir / "candidates" / f"{cid}.joblib"
            joblib.dump(model, path)
            r = {"id": cid, "status": "done", "fit_seconds": round(fit_s, 3), "val": {**rep, **extra}, "artifact": str(path), "artifact_sha256": sha256_file(path)}
            results_out.append(r)
            ev("candidate_done", **r)

        if base is not None and base.covered is not None:
            rep = make_report(base, base.oof_pred, base.oof_proba, base.covered)
            bm = pl.full_pipeline(vr.feature_kinds, zoo.build("baseline", {}, task, plan["seed"]))
            bm.fit(Xd, yd)
            save("baseline", bm, base.fit_seconds, rep, {"cv_mean": base.cv_mean, "cv_std": base.cv_std})
        for r in finals:
            clock.check(hard=False)
            t0 = time.perf_counter()
            m = pl.full_pipeline(vr.feature_kinds, zoo.build(r.model_id, r.params, task, plan["seed"], plan.get("n_jobs", 1)))
            m.fit(Xd, yd)
            fit_s = time.perf_counter() - t0
            models[r.model_id] = m
            rep = make_report(r, r.oof_pred, r.oof_proba, r.covered)
            train_pred = m.predict(Xd)
            train_proba = m.predict_proba(Xd) if task == "classification" and hasattr(m, "predict_proba") else None
            train_score = mx.score(metric, yd, train_pred, train_proba, classes)
            r.train_score = train_score
            save(r.model_id, m, fit_s, rep, {"cv_mean": r.cv_mean, "cv_std": r.cv_std, "fold_scores": r.fold_scores, "tuned": r.tuned,
                                             "n_trials": r.n_trials, "default_cv": r.default_mean, "params": {k: _py(v) for k, v in r.params.items()},
                                             "train_score": train_score})
        if ens and ens["accepted"]:
            members = [(mid, w) for mid, w in ens["members"]]
            ests = [(mid, pl.full_pipeline(vr.feature_kinds, zoo.build(mid, next(f.params for f in finals if f.model_id == mid), task, plan["seed"],
                                                                      plan.get("n_jobs", 1)))) for mid, _ in members]
            ws = [w for _, w in members]
            em = VotingClassifier(ests, voting="soft", weights=ws) if task == "classification" else VotingRegressor(ests, weights=ws)
            t0 = time.perf_counter()
            em.fit(Xd, yd)
            fit_s = time.perf_counter() - t0
            rep = make_report(ens, ens["oof_pred"], ens["oof_proba"], ens["covered"])
            ev("candidate_add", id="ensemble", label="Weighted ensemble", estimator="ensemble", params={"members": members}, description="Blend of the best models.")
            save("ensemble", em, fit_s, rep, {"cv_mean": ens["score"], "cv_std": finals[0].cv_std, "members": members})
            ev("ensemble", members=members, gain=ens["gain"], margin=ens["margin"], accepted=True, score=ens["score"], best_single=ens["best_single"])
        elif ens:
            ev("ensemble", members=ens["members"], gain=ens["gain"], margin=ens["margin"], accepted=False, score=ens["score"], best_single=ens["best_single"])

        # ---- choose
        sel_id, reason = choose(finals, ens, profile_name)
        sel_result = next((r for r in finals if r.model_id == sel_id), finals[0])
        sel_cv = ens["score"] if sel_id == "ensemble" else sel_result.cv_mean
        sel_std = sel_result.cv_std
        # "meaningfully better than doing nothing": more than one fold-to-fold standard deviation and at least a small absolute/relative gain
        floor = 0.02 * abs(base_mean) if (base_mean is not None and metric in mx.LOWER_IS_BETTER) else 0.02
        margin = max(floor, sel_std)
        beats = None if base_mean is None else bool(sel_cv - base_mean > margin)

        # ---- diagnose (learning curve on the chosen single model, if time allows)
        curve = []
        if clock.left() > -0.5 * budget:  # the curve has its own small allowance; skip only if the search already ran far over budget
            sub_clock = sr.Clock(max(5.0, 0.12 * budget), cancelled)
            curve = diagnose.learning_curve(sel_result.model_id, sel_result.params, Xd, yd, cfg, sub_clock)
        per_class = None
        sel_rep = next(r["val"] for r in results_out if r["id"] == sel_id)
        if task == "classification":
            per_class = sel_rep.get("per_class")
        recs = diagnose.recommend(task=task, metric=metric, best={"cv_mean": sel_cv, "cv_std": sel_std, "fold_scores": sel_result.fold_scores},
                                  baseline_mean=base_mean, curve=curve, train_score=getattr(sel_result, "train_score", None), per_class=per_class,
                                  class_counts={k: v for k, v in counts["counts"].items()}, profile={"columns": [vars(p) for p in profiles]},
                                  findings=all_findings, n_dev=len(dev), cv_scheme=scheme)
        ev("recommendations", items=recs)

        ev("selected", id=sel_id, metric=metric, beats_baseline=beats, reason=reason)
        summary = {"status": "finished", "mode": "automl", "selected": sel_id, "selection_metric": metric, "higher_is_better": higher,
                   "beats_baseline": beats, "selection_reason": reason, "split_hash": sh, "split_strategy": strategy, "cv_scheme": scheme,
                   "classes": classes, "feature_kinds": vr.feature_kinds, "n_rows_used": int(len(rows)), "n_dev": int(len(dev)),
                   "budget_seconds": budget, "seconds_used": round(time.perf_counter() - t_start, 1), "search_log": res["log"],
                   "ensemble": None if not ens else {k: ens[k] for k in ("members", "gain", "margin", "accepted", "score", "best_single")},
                   "learning_curve": curve, "recommendations": recs, "candidates": results_out,
                   "advice": plan.get("advice")}
        (run_dir / "summary.json").write_text(json.dumps(summary, default=str, indent=1), encoding="utf-8")
        ev("finished", summary=summary)
        return summary
    except sr.Cancelled:
        ev("cancelled")
        return {"status": "cancelled"}
    except (DatasetError, SplitError) as e:
        ev("failed", reason=str(e))
        return {"status": "failed"}
    except Exception as e:
        ev("failed", reason=f"Unexpected error: {type(e).__name__}: {e}", trace=traceback.format_exc())
        return {"status": "failed"}
