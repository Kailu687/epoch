"""Experiment execution. `run_plan` is a pure, process-agnostic function: it reads a JSON-serialisable plan,
trains candidates on the TRAIN split, scores them on VALIDATION, and never touches the test split."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Callable

import joblib
import numpy as np

from . import pipeline as pl
from .candidates import provider_for
from .evaluate import evaluate, primary_metric
from .io_csv import DatasetError, read_csv, sha256_file
from .profile import class_balance, profile_columns
from .splits import SplitError, make_split, split_hash
from .validate import leakage_probe, validate

Emit = Callable[[dict], None]


class Cancelled(Exception):
    pass


def _py(v):
    return v.item() if hasattr(v, "item") else v


def run_plan(plan: dict, emit: Emit, cancelled: Callable[[], bool] = lambda: False) -> dict:
    if plan.get("mode") == "automl":  # v2 search engine (cross-validation, tuning, ensembling)
        from .automl.engine import run_automl
        return run_automl(plan, emit, cancelled)
    run_dir = Path(plan["run_dir"])
    (run_dir / "candidates").mkdir(parents=True, exist_ok=True)
    ev = lambda type_, **kw: emit({"type": type_, "ts": time.time(), **kw})
    task = plan["task_type"]
    try:
        ev("stage", name="loading")
        df, info = read_csv(plan["dataset_path"])
        if plan.get("dataset_sha256") and info.sha256 != plan["dataset_sha256"]:
            raise DatasetError("Dataset file changed since import (checksum mismatch).")
        profiles = profile_columns(df)
        ev("stage", name="validating")
        vr = validate(df, profiles, plan["target"], task, plan.get("exclude"), plan.get("include"), plan.get("drop_duplicates", False))
        ev("findings", findings=[f.to_dict() for f in vr.findings])
        if not vr.ok:
            ev("failed", reason="Validation errors: " + " | ".join(f.message for f in vr.errors))
            return {"status": "failed"}

        rows = np.flatnonzero(vr.row_mask)
        X = pl.prepare_frame(df.iloc[rows], vr.feature_kinds).reset_index(drop=True)
        y_raw = df[plan["target"]].iloc[rows].reset_index(drop=True)
        if task == "classification":
            classes = sorted(y_raw.unique().tolist(), key=lambda v: (str(type(v)), v))
            classes = [_py(c) for c in classes]
            y = np.array([_py(v) for v in y_raw], dtype=object if any(isinstance(c, str) for c in classes) else None)
            imbalance = class_balance(y_raw)["imbalance_ratio"]
        else:
            classes, y, imbalance = None, y_raw.to_numpy(dtype="float64"), None

        ev("stage", name="splitting")
        split = make_split(y, task, plan["seed"], plan.get("val_frac", 0.15), plan.get("test_frac", 0.15))
        sh = split_hash(split)
        (run_dir / "split.json").write_text(json.dumps({
            "seed": plan["seed"], "hash": sh, "indices_refer_to": "rows kept after cleaning; original_rows maps them to CSV data rows (0-based)",
            "original_rows": rows.tolist(), **{k: v.tolist() for k, v in split.items()}}), encoding="utf-8")
        ev("split", sizes={k: int(len(v)) for k, v in split.items()}, hash=sh)

        tr, va = split["train"], split["val"]
        Xtr, ytr, Xva, yva = X.iloc[tr], y[tr], X.iloc[va], y[va]

        ev("stage", name="leakage-check")
        leak = leakage_probe(Xtr, ytr, vr.feature_kinds, task, plan["seed"])
        if leak:
            ev("findings", findings=[f.to_dict() for f in leak])

        provider = provider_for(task)
        specs = provider.candidates(task, len(tr), imbalance, plan["seed"], plan.get("n_jobs", 1))
        want = plan.get("candidates")
        if want:
            specs = [s for s in specs if s.id in want or s.is_baseline]
        ev("plan", candidates=[{"id": s.id, "label": s.label, "estimator": s.estimator, "params": s.params, "is_baseline": s.is_baseline,
                                "description": s.description} for s in specs])
        metric, higher = primary_metric(task)
        results = []
        for spec in specs:
            if cancelled():
                raise Cancelled()
            ev("candidate_start", id=spec.id)
            t0 = time.perf_counter()
            try:
                model = pl.full_pipeline(vr.feature_kinds, provider.build(spec))
                model.fit(Xtr, ytr)
                fit_s = time.perf_counter() - t0
                rep = evaluate(model, Xva, yva, task, classes)
                path = run_dir / "candidates" / f"{spec.id}.joblib"
                joblib.dump(model, path)
                r = {"id": spec.id, "status": "done", "fit_seconds": round(fit_s, 3), "val": rep, "artifact": str(path),
                     "artifact_sha256": sha256_file(path)}
                ev("candidate_done", **r)
            except Exception as e:  # a failing candidate must not abort the run
                r = {"id": spec.id, "status": "failed", "error": f"{type(e).__name__}: {e}"}
                ev("candidate_failed", **r)
            results.append(r)

        done = [r for r in results if r["status"] == "done"]
        if not done:
            ev("failed", reason="Every candidate failed.")
            return {"status": "failed"}
        key = lambda r: (-r["val"][metric] if higher else r["val"][metric], r["fit_seconds"])
        best = min(done, key=key)
        base = next((r for r in done if r["id"] == "baseline"), None)
        beats = None if base is None or best["id"] == "baseline" else (
            best["val"][metric] > base["val"][metric] if higher else best["val"][metric] < base["val"][metric])
        summary = {"status": "finished", "selected": best["id"], "selection_metric": metric, "higher_is_better": higher,
                   "beats_baseline": beats, "split_hash": sh, "classes": classes, "feature_kinds": vr.feature_kinds,
                   "n_rows_used": int(len(rows)), "candidates": results}
        (run_dir / "summary.json").write_text(json.dumps(summary, default=str, indent=1), encoding="utf-8")
        ev("selected", id=best["id"], metric=metric, beats_baseline=beats)
        ev("finished", summary=summary)
        return summary
    except Cancelled:
        ev("cancelled")
        return {"status": "cancelled"}
    except (DatasetError, SplitError) as e:
        ev("failed", reason=str(e))
        return {"status": "failed"}
    except Exception as e:
        ev("failed", reason=f"Unexpected error: {type(e).__name__}: {e}", trace=traceback.format_exc())
        return {"status": "failed"}
