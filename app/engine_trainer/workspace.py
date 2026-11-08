"""Workspace: the application service used by both the GUI and the CLI."""
from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path

import joblib
import numpy as np

from . import pipeline as pl
from .evaluate import evaluate, primary_metric
from .export import env_versions, write_bundle
from .io_csv import read_csv
from .jobs import Job
from .profile import infer_task, profile_columns, profile_dataset
from .store import Store, now


class WorkspaceError(Exception):
    pass


def _id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)[:80] or "dataset"


class Workspace:
    def __init__(self, path):
        self.root = Path(path).resolve()
        for d in ("datasets", "runs", "models", "exports"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        legacy = self.root / "engine_trainer.sqlite"  # workspaces created before the rename to Epoch
        self.store = Store(legacy if legacy.exists() else self.root / "epoch.sqlite")
        self._cache = None  # (dataset_id, DataFrame)

    def close(self):
        self.store.close()

    # ---------------------------------------------------------------- datasets
    @staticmethod
    def prepare_import(csv_path) -> dict:
        """Thread-safe, DB-free half of an import: parse and profile. Safe to run in a background thread."""
        df, info = read_csv(csv_path)  # raises DatasetError with a user-facing message
        return {"df": df, "info": info, "profile": profile_dataset(df), "csv_path": str(csv_path)}

    def commit_import(self, prep: dict) -> str:
        """Main-thread half: copy the file into the workspace and record it (owns the SQLite connection)."""
        csv_path, df, info = prep["csv_path"], prep["df"], prep["info"]
        did = _id("ds")
        ddir = self.root / "datasets" / did
        ddir.mkdir()
        stored = ddir / _safe_name(csv_path)
        shutil.copy2(csv_path, stored)  # immutable copy for provenance
        (ddir / "profile.json").write_text(json.dumps(prep["profile"], indent=1), encoding="utf-8")
        self.store.insert("datasets", id=did, name=Path(csv_path).name, source_path=str(Path(csv_path).resolve()), stored_path=str(stored),
                          sha256=info.sha256, domain="tabular", n_rows=len(df), n_cols=df.shape[1], profile_json=prep["profile"], imported_at=now())
        self._cache = (did, df)
        return did

    def import_dataset(self, csv_path) -> str:
        return self.commit_import(self.prepare_import(csv_path))

    def _df(self, did):
        """Parsed DataFrame of a stored dataset; the most recent one is cached to keep the UI responsive."""
        if self._cache and self._cache[0] == did:
            return self._cache[1]
        df, _ = read_csv(self.dataset(did)["stored_path"])
        self._cache = (did, df)
        return df

    def dataset(self, did) -> dict:
        d = self.store.one("SELECT * FROM datasets WHERE id=?", did)
        if not d:
            raise WorkspaceError(f"Unknown dataset {did}")
        d["profile"] = json.loads(d["profile_json"])
        return d

    def datasets(self):
        return self.store.all("SELECT id,name,n_rows,n_cols,sha256,imported_at FROM datasets ORDER BY imported_at DESC")

    def preview(self, did, n=50):
        return self._df(did).head(n)

    def suggest_task(self, did, target):
        return infer_task(self._df(did), target)

    def validate_preview(self, did, target, task_type, exclude=None, drop_duplicates=False):
        """Run structural validation without training (no leakage probe; that runs on training rows during the experiment)."""
        from .validate import validate
        df = self._df(did)
        return validate(df, profile_columns(df), target, task_type, exclude, None, drop_duplicates)

    # ---------------------------------------------------------------- advisor ("Let Epoch decide")
    def advise(self, did, description: str = "", hints: dict | None = None):
        """Plain-language description + dataset -> Advice (task, target, metric, split, budget, exclusions, explanations, questions)."""
        from .advisor import advise
        return advise(description, self.dataset(did)["profile"], self._df(did), hints)

    def create_auto_experiment(self, did, advice, *, seed=42, **overrides) -> str:
        """Create an AutoML v2 experiment from a resolved Advice. `overrides` may set budget_mode, budget_seconds, optimisation_profile, etc."""
        a = advice.to_dict() if hasattr(advice, "to_dict") else dict(advice)
        if a.get("task_type") not in ("classification", "regression") or not a.get("target"):
            raise WorkspaceError("Epoch still needs the target column and task type: answer its questions first.")
        opts = dict(mode="automl", metric=a.get("metric"), split_strategy=a.get("split_strategy"), group_column=a.get("group_column"),
                    time_column=a.get("time_column"), budget_mode=a.get("budget_mode"), optimisation_profile=a.get("optimisation_profile"), advice=a)
        opts.update({k: v for k, v in overrides.items() if v is not None})
        return self.create_experiment(did, target=a["target"], task_type=a["task_type"], seed=seed, exclude=a.get("exclude_columns") or [], **opts)

    # ---------------------------------------------------------------- try a model / report
    def try_model(self, model_id, data):
        """Predict with a registered model. `data` is a DataFrame, a CSV path or a dict (one row). Returns a DataFrame with prediction columns."""
        import pandas as pd
        from .inference import prepare_input
        m = self.store.one("SELECT * FROM models WHERE id=?", model_id)
        if not m:
            raise WorkspaceError(f"Unknown model {model_id}")
        exp = self.experiment(m["experiment_id"])
        summary = json.loads((Path(exp["run_dir"]) / "summary.json").read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif not isinstance(data, pd.DataFrame):
            if not str(data).lower().endswith((".csv", ".txt", ".tsv")):
                raise WorkspaceError("Provide a CSV file, a DataFrame or a dict.")
            data, _ = read_csv(data)
        schema = {"columns": [{"name": c, "kind": k} for c, k in summary["feature_kinds"].items()]}
        try:
            X = prepare_input(data, schema)
        except ValueError as e:
            raise WorkspaceError(str(e)) from e
        model = joblib.load(self._verified_artifact(m))
        out = pd.DataFrame(index=data.index)
        out["prediction"] = model.predict(X)
        if exp["task"]["task_type"] == "classification":
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                out["confidence"] = proba.max(axis=1)
                for i, c in enumerate(model.classes_):
                    out[f"p({c})"] = proba[:, i]
        return out

    def report_html(self, model_id) -> str:
        from .report import build_report_html
        m = self.store.one("SELECT * FROM models WHERE id=?", model_id)
        exp = self.experiment(m["experiment_id"])
        summary = json.loads((Path(exp["run_dir"]) / "summary.json").read_text(encoding="utf-8"))
        cand = next(c for c in exp["candidates"] if c["id"] == m["candidate_id"])
        test = json.loads(m["test_metrics_json"]) if m["test_metrics_json"] else None
        ds = self.store.one("SELECT name FROM datasets WHERE id=?", exp["dataset_id"])
        return build_report_html(exp, summary, cand["cand_key"], test, m["test_eval_count"] or 0, json.loads(exp["env_json"]), ds["name"] if ds else "")

    def candidate_specs(self, did, target, task_type):
        from .candidates import provider_for
        df = self._df(did)
        return provider_for(task_type).candidates(task_type, len(df), None, 0, 1)

    # ---------------------------------------------------------------- experiments
    def create_experiment(self, did, *, target, task_type, seed=42, val_frac=0.15, test_frac=0.15, exclude=None, include=None,
                          drop_duplicates=False, candidates=None, n_jobs=None, **options) -> str:
        """`options` (AutoML v2): mode='automl', budget_mode, budget_seconds, optimisation_profile, metric, split_strategy,
        group_column, time_column, cv_folds, models, max_trials, advice."""
        if task_type not in ("classification", "regression"):
            raise WorkspaceError("task_type must be 'classification' or 'regression'")
        ds = self.dataset(did)
        eid = _id("exp")
        run_dir = self.root / "runs" / eid
        run_dir.mkdir()
        plan = {"experiment_id": eid, "run_dir": str(run_dir), "dataset_path": ds["stored_path"], "dataset_sha256": ds["sha256"],
                "target": target, "task_type": task_type, "seed": int(seed), "val_frac": val_frac, "test_frac": test_frac,
                "exclude": exclude or [], "include": include or [], "drop_duplicates": drop_duplicates, "candidates": candidates,
                "n_jobs": n_jobs or max(1, (os.cpu_count() or 2) - 1)}
        plan.update({k: v for k, v in options.items() if v is not None})
        (run_dir / "config.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
        self.store.insert("experiments", id=eid, dataset_id=did, task_json=plan, seed=int(seed), status="created", env_json=env_versions(),
                          run_dir=str(run_dir), created_at=now())
        return eid

    def start(self, eid) -> Job:
        exp = self.experiment(eid)
        self.store.update("experiments", eid, status="running")
        return Job(exp["task"]).start()

    def handle_event(self, eid, ev) -> None:
        """Persist an event from a Job. Call from the thread that owns the Workspace."""
        t = ev["type"]
        if t in ("resource", "trial", "screened", "validated", "search_plan"):
            return
        if t == "findings":
            old = self.store.one("SELECT findings_json FROM experiments WHERE id=?", eid)["findings_json"]
            allf = (json.loads(old) if old else []) + ev["findings"]
            self.store.update("experiments", eid, findings_json=allf)
        elif t == "split":
            self.store.update("experiments", eid, split_json={"sizes": ev["sizes"], "hash": ev["hash"]})
        elif t == "plan":
            for c in ev["candidates"]:
                self.store.insert("candidates", id=f"{eid}:{c['id']}", experiment_id=eid, cand_key=c["id"], provider="sklearn-tabular",
                                  estimator=c["estimator"], params_json=c["params"], status="queued")
        elif t == "candidate_add":
            self.store.insert("candidates", id=f"{eid}:{ev['id']}", experiment_id=eid, cand_key=ev["id"], provider="epoch-automl",
                              estimator=ev["estimator"], params_json=ev["params"], status="queued")
        elif t == "candidate_skipped":
            self.store.update("candidates", f"{eid}:{ev['id']}", status="skipped", error=ev.get("reason"))
        elif t == "candidate_screened":
            cid = f"{eid}:{ev['id']}"
            self.store.update("candidates", cid, status="screened", fit_seconds=ev.get("seconds"),
                              val_metrics_json={"screened_only": True, "cv_mean": ev["cv_mean"], "cv_std": ev["cv_std"], "score": ev.get("score")})
        elif t == "candidate_start":
            self.store.update("candidates", f"{eid}:{ev['id']}", status="running")
        elif t == "candidate_done":
            cid = f"{eid}:{ev['id']}"
            self.store.update("candidates", cid, status="done", fit_seconds=ev["fit_seconds"], val_metrics_json=ev["val"])
            if ev.get("artifact"):
                self.store.insert("models", id=f"mdl-{eid[4:]}-{ev['id']}", candidate_id=cid, experiment_id=eid, artifact_path=ev["artifact"],
                                  artifact_sha256=ev["artifact_sha256"], format="joblib-sklearn-pipeline", created_at=now())
        elif t == "candidate_failed":
            self.store.update("candidates", f"{eid}:{ev['id']}", status="failed", error=ev["error"])
        elif t == "selected":
            self.store.update("models", f"mdl-{eid[4:]}-{ev['id']}", selected=1)
        elif t == "finished":
            self.store.update("experiments", eid, status="finished", finished_at=now())
        elif t == "failed":
            self.store.update("experiments", eid, status="failed", error=ev.get("reason"), finished_at=now())
        elif t == "cancelled":
            self.store.update("experiments", eid, status="cancelled", finished_at=now())
        self.store.log(eid, "error" if t == "failed" else "info", t, {k: v for k, v in ev.items() if k not in ("summary", "val", "findings")})

    def run_blocking(self, eid, on_event=None):
        job = self.start(eid)

        def cb(e):
            self.handle_event(eid, e)
            if on_event:
                on_event(e)
        return job.wait(cb)

    def experiment(self, eid) -> dict:
        e = self.store.one("SELECT * FROM experiments WHERE id=?", eid)
        if not e:
            raise WorkspaceError(f"Unknown experiment {eid}")
        e["task"] = json.loads(e["task_json"])
        e["findings"] = json.loads(e["findings_json"]) if e["findings_json"] else []
        e["split"] = json.loads(e["split_json"]) if e["split_json"] else None
        e["candidates"] = self.store.all("SELECT * FROM candidates WHERE experiment_id=? ORDER BY rowid", eid)
        for c in e["candidates"]:
            c["val"] = json.loads(c["val_metrics_json"]) if c["val_metrics_json"] else None
        e["models"] = self.store.all("SELECT * FROM models WHERE experiment_id=?", eid)
        return e

    def experiments(self):
        return self.store.all("SELECT e.id, e.status, e.created_at, e.task_json, d.name AS dataset FROM experiments e "
                              "JOIN datasets d ON d.id=e.dataset_id ORDER BY e.created_at DESC")

    # ---------------------------------------------------------------- test set (single use)
    def evaluate_test(self, model_id, *, acknowledge_repeat=False) -> dict:
        """Evaluate a model on the sealed test split. The first call is recorded; repeats need acknowledgement."""
        m = self.store.one("SELECT * FROM models WHERE id=?", model_id)
        if not m:
            raise WorkspaceError(f"Unknown model {model_id}")
        if m["test_eval_count"] and not acknowledge_repeat:
            raise WorkspaceError("The test set has already been used for this model. Repeated use biases the estimate; "
                                 "pass acknowledge_repeat=True to proceed and record it.")
        exp = self.experiment(m["experiment_id"])
        plan = exp["task"]
        summary = json.loads((Path(exp["run_dir"]) / "summary.json").read_text(encoding="utf-8"))
        df, _ = read_csv(plan["dataset_path"])
        sp = json.loads((Path(exp["run_dir"]) / "split.json").read_text(encoding="utf-8"))
        rows = np.array(sp["original_rows"])
        kinds = summary["feature_kinds"]
        X = pl.prepare_frame(df.iloc[rows], kinds).reset_index(drop=True)
        y_raw = df[plan["target"]].iloc[rows].reset_index(drop=True)
        te = np.array(sp["test"])
        model = joblib.load(self._verified_artifact(m))
        if plan["task_type"] == "classification":
            y = np.array(y_raw.tolist(), dtype=object if any(isinstance(c, str) for c in summary["classes"]) else None)
        else:
            y = y_raw.to_numpy(dtype="float64")
        rep = evaluate(model, X.iloc[te], y[te], plan["task_type"], summary["classes"])
        rep["split_hash"] = sp["hash"]
        self.store.update("models", model_id, test_metrics_json=rep, test_eval_count=(m["test_eval_count"] or 0) + 1, test_consumed_at=now())
        self.store.log(m["experiment_id"], "warning" if m["test_eval_count"] else "info", "test_evaluation", {"model": model_id, "count": (m["test_eval_count"] or 0) + 1})
        return rep

    def _verified_artifact(self, m):
        from .io_csv import sha256_file
        p = Path(m["artifact_path"])
        if not p.is_file() or sha256_file(p) != m["artifact_sha256"]:
            raise WorkspaceError("Model artifact is missing or does not match its recorded checksum; refusing to load it.")
        return p

    # ---------------------------------------------------------------- registry/export
    def summary(self, eid) -> dict | None:
        """The run's summary.json (None if the run has not finished)."""
        p = Path(self.experiment(eid)["run_dir"]) / "summary.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    # ---------------------------------------------------------------- improve an existing model with more data
    def check_new_data(self, model_id, new_dataset_id) -> dict:
        """Compatibility and drift report for adding `new_dataset_id` to the model's training data."""
        from .improve import check_compatibility
        m = self.store.one("SELECT * FROM models WHERE id=?", model_id)
        if not m:
            raise WorkspaceError(f"Unknown model {model_id}")
        exp = self.experiment(m["experiment_id"])
        summ = self.summary(exp["id"])
        old_df = self._df(exp["dataset_id"])
        new_df = self._df(new_dataset_id)
        return check_compatibility(old_df, new_df, summ["feature_kinds"], exp["task"]["target"], exp["task"]["task_type"], summ["classes"])

    def improve_model(self, model_id, new_dataset_id, *, budget_mode=None, budget_seconds=None, holdout_frac=0.15) -> str:
        """Create an experiment that retrains on old + new rows. The ORIGINAL locked test rows stay the test set, so the old and new
        models are compared on exactly the same data; a slice of the new rows is held out as a second, separate test."""
        import pandas as pd
        rep = self.check_new_data(model_id, new_dataset_id)
        if not rep["ok"]:
            raise WorkspaceError(rep["errors"][0])
        m = self.store.one("SELECT * FROM models WHERE id=?", model_id)
        old = self.experiment(m["experiment_id"])
        sp = json.loads((Path(old["run_dir"]) / "split.json").read_text(encoding="utf-8"))
        old_df = self._df(old["dataset_id"])
        new_df = self._df(new_dataset_id)
        cols = list(old_df.columns)
        combined = pd.concat([old_df[cols], new_df[cols]], ignore_index=True)
        out = self.root / "datasets" / "_combined"
        out.mkdir(exist_ok=True)
        old_name = self.store.one("SELECT name FROM datasets WHERE id=?", old["dataset_id"])["name"]
        new_name = self.store.one("SELECT name FROM datasets WHERE id=?", new_dataset_id)["name"]
        path = out / f"{Path(old_name).stem}+{Path(new_name).stem}_{_id('c')[2:]}.csv"
        combined.to_csv(path, index=False)
        did = self.import_dataset(path)
        test_rows = [sp["original_rows"][i] for i in sp["test"]]
        plan_opts = dict(old["task"])
        eid = self.create_experiment(
            did, target=plan_opts["target"], task_type=plan_opts["task_type"], seed=plan_opts["seed"], exclude=plan_opts.get("exclude"),
            drop_duplicates=plan_opts.get("drop_duplicates", False), n_jobs=plan_opts.get("n_jobs"), mode="automl", metric=plan_opts.get("metric"),
            split_strategy=plan_opts.get("split_strategy"), group_column=plan_opts.get("group_column"), time_column=plan_opts.get("time_column"),
            budget_mode=budget_mode or plan_opts.get("budget_mode"), budget_seconds=budget_seconds or plan_opts.get("budget_seconds"),
            optimisation_profile=plan_opts.get("optimisation_profile"), models=plan_opts.get("models"), advice=plan_opts.get("advice"),
            continue_from={"experiment_id": old["id"], "old_model_id": model_id, "n_old_rows": len(old_df), "old_test_rows": test_rows, "holdout_frac": holdout_frac})
        return eid

    def _score_on_rows(self, model_id, frame, rows):
        """Evaluate a registered model on given CSV rows of `frame` (prepared with the model's own feature kinds)."""
        from . import pipeline as pl
        m = self.store.one("SELECT * FROM models WHERE id=?", model_id)
        exp = self.experiment(m["experiment_id"])
        summ = self.summary(exp["id"])
        plan = exp["task"]
        sub = frame.iloc[rows]
        X = pl.prepare_frame(sub, summ["feature_kinds"]).reset_index(drop=True)
        y_raw = sub[plan["target"]].reset_index(drop=True)
        y = np.array(y_raw.tolist(), dtype=object if any(isinstance(c, str) for c in (summ["classes"] or [])) else None) if plan["task_type"] == "classification" \
            else y_raw.to_numpy(dtype="float64")
        if plan["task_type"] == "classification":
            keep = np.isin(y, summ["classes"])  # classes the old model cannot know are skipped (and counted by the caller)
        else:
            keep = np.ones(len(y), bool)
        model = joblib.load(self._verified_artifact(m))
        rep = evaluate(model, X[keep], y[keep], plan["task_type"], summ["classes"])
        rep["n_unknown_class_rows"] = int((~keep).sum())
        return rep

    def compare_improvement(self, old_model_id, new_model_id) -> dict:
        """Old versus new model on (1) the original locked test rows and (2) the held-out slice of the new data.
        Uses the original test set once more for the new model (recorded), and evaluates the old model there only if that was not done yet."""
        new_m = self.store.one("SELECT * FROM models WHERE id=?", new_model_id)
        new_exp = self.experiment(new_m["experiment_id"])
        cf = new_exp["task"].get("continue_from")
        if not cf:
            raise WorkspaceError("That model was not created by 'improve': nothing to compare.")
        sp = json.loads((Path(new_exp["run_dir"]) / "split.json").read_text(encoding="utf-8"))
        old_m = self.store.one("SELECT * FROM models WHERE id=?", old_model_id)
        summ = self.summary(new_exp["id"])
        metric = summ["selection_metric"]
        lower = metric in ("rmse", "mae", "log_loss")
        old_test = json.loads(old_m["test_metrics_json"]) if old_m["test_metrics_json"] else self.evaluate_test(old_model_id)
        new_test = json.loads(new_m["test_metrics_json"]) if new_m["test_metrics_json"] else self.evaluate_test(new_model_id)
        frame = self._df(new_exp["dataset_id"])
        hold_rows = [sp["original_rows"][i] for i in sp["new_holdout"]]
        old_hold = self._score_on_rows(old_model_id, frame, hold_rows)
        new_hold = self._score_on_rows(new_model_id, frame, hold_rows)
        key = metric if metric in new_test else ("macro_f1" if "macro_f1" in new_test else "rmse")
        lower = key in ("rmse", "mae", "log_loss")
        sign = -1 if lower else 1
        d_test = sign * (new_test[key] - old_test[key])
        d_hold = sign * (new_hold[key] - old_hold[key])
        tol = 0.01 if not lower else 0.01 * abs(old_test[key])
        if d_test > tol and d_hold > -tol:
            verdict, why = "adopt", "The new model is better on the original test data and not worse on the new data."
        elif d_test < -tol or d_hold < -tol:
            verdict, why = "keep_old", "The new model is worse on at least one test set. Adding this data did not help."
        else:
            verdict, why = "inconclusive", "The differences are within normal variation: neither model is clearly better."
        return {"metric": key, "old_test": old_test, "new_test": new_test, "old_new_data": old_hold, "new_new_data": new_hold,
                "delta_original_test": d_test, "delta_new_data": d_hold, "verdict": verdict, "explanation": why,
                "note": "Positive deltas mean the new model is better. Both test sets are small; treat differences of about 1 point or less as noise."}

    def models(self):
        return self.store.all("SELECT m.id, m.experiment_id, m.selected, m.test_eval_count, c.cand_key, c.estimator, c.val_metrics_json, m.created_at "
                              "FROM models m JOIN candidates c ON c.id=m.candidate_id ORDER BY m.created_at DESC")

    def selected_model(self, eid) -> str | None:
        r = self.store.one("SELECT id FROM models WHERE experiment_id=? AND selected=1", eid)
        return r["id"] if r else None

    def export(self, model_id, dest) -> dict:
        m = self.store.one("SELECT * FROM models WHERE id=?", model_id)
        if not m:
            raise WorkspaceError(f"Unknown model {model_id}")
        exp = self.experiment(m["experiment_id"])
        summary = json.loads((Path(exp["run_dir"]) / "summary.json").read_text(encoding="utf-8"))
        cand = next(c for c in exp["candidates"] if c["id"] == m["candidate_id"])
        df, _ = read_csv(exp["task"]["dataset_path"])
        profs = {p.name: p for p in profile_columns(df)}
        cols = [{"name": c, "kind": k, "example": (profs[c].example_values[0] if profs[c].example_values else None)} for c, k in summary["feature_kinds"].items()]
        ex = {}
        for c in cols:
            if c["kind"] == "numeric":
                try:
                    ex[c["name"]] = float(c["example"])
                except (TypeError, ValueError):
                    ex[c["name"]] = None
            else:
                ex[c["name"]] = c["example"]
        report = {"task_type": exp["task"]["task_type"], "selection_metric": summary["selection_metric"], "model": cand["cand_key"],
                  "validation": cand["val"], "test": json.loads(m["test_metrics_json"]) if m["test_metrics_json"] else None,
                  "test_note": None if m["test_metrics_json"] else "Test set not evaluated for this model.",
                  "test_evaluations": m["test_eval_count"], "beats_baseline_on_validation": summary["beats_baseline"],
                  "baseline_validation": next((c["val"] for c in exp["candidates"] if c["cand_key"] == "baseline"), None),
                  "split": exp["split"], "findings": exp["findings"],
                  "limitations": ["Estimates come from a single split and may vary with a different seed.",
                                  "Performance on data unlike the training data is unknown."]}
        config = {"experiment_id": exp["id"], "plan": exp["task"], "estimator": cand["estimator"], "params": json.loads(cand["params_json"]),
                  "dataset_sha256": exp["task"]["dataset_sha256"], "environment": json.loads(exp["env_json"]), "created_at": exp["created_at"]}
        manifest = write_bundle(Path(dest), model_path=self._verified_artifact(m), schema_cols=cols, classes=summary["classes"], config=config,
                                report=report, env=env_versions(), example_row=ex, extra_files={"report.html": self.report_html(model_id)})
        self.store.insert("exports", id=_id("exp-out"), model_id=model_id, path=str(dest), manifest_json=manifest, created_at=now())
        return manifest
