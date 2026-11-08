"""Command-line interface: headless equivalent of the GUI workflow."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .io_csv import DatasetError
from .workspace import Workspace, WorkspaceError


def _fmt_event(e):
    t = e["type"]
    if t == "stage":
        return f"[stage] {e['name']}"
    if t == "findings":
        return "\n".join(f"[{f['severity']}] {f['message']}" for f in e["findings"])
    if t == "split":
        return f"[split] {e['sizes']}"
    if t == "candidate_start":
        return f"[train] {e['id']} ..."
    if t == "candidate_done":
        v = e["val"]
        k = "macro_f1" if "macro_f1" in v else "rmse"
        extra = f" cv={v['cv_mean']:.4f}±{v.get('cv_std', 0):.4f}" if "cv_mean" in v else ""
        return f"[done ] {e['id']}: {k}={v.get(k, float('nan')):.4f}{extra} ({e['fit_seconds']}s)"
    if t == "screened":
        return f"[screen] {e['id']}: cv={e['cv_mean']:.4f} ({e['seconds']}s)"
    if t == "validated":
        return f"[tuned ] {e['id']}: cv={e['cv_mean']:.4f}±{e['cv_std']:.4f}"
    if t == "ensemble":
        return f"[ensemble] {'kept' if e['accepted'] else 'not kept'}: gain {e['gain']:+.4f} (needs > {e['margin']:.4f})"
    if t == "recommendations":
        return "\n".join(f"[advice:{r['severity']}] {r['title']}" for r in e["items"])
    if t == "candidate_failed":
        return f"[FAIL ] {e['id']}: {e['error']}"
    if t == "selected":
        return f"[select] {e['id']} (beats baseline on validation: {e['beats_baseline']})"
    if t in ("failed", "cancelled"):
        return f"[{t}] {e.get('reason', '')}"
    return None


def cmd_train(a) -> int:
    ws = Workspace(a.workspace or tempfile.mkdtemp(prefix="et_ws_"))
    try:
        did = ws.import_dataset(a.csv)
        ds = ws.dataset(did)
        if a.target not in [c["name"] for c in ds["profile"]["columns"]]:
            print(f"Target '{a.target}' not found. Columns: {[c['name'] for c in ds['profile']['columns']]}", file=sys.stderr)
            return 2
        task = a.task
        if task == "auto":
            s = ws.suggest_task(did, a.target)
            task = s.task_type
            print(f"[task] inferred {task} ({s.confidence}): {'; '.join(s.reasons)}")
        eid = ws.create_experiment(did, target=a.target, task_type=task, seed=a.seed, exclude=a.exclude, drop_duplicates=a.dedupe)
        ws.run_blocking(eid, lambda e: (lambda s: print(s) if s else None)(_fmt_event(e)))
        exp = ws.experiment(eid)
        if exp["status"] != "finished":
            print(f"Run ended with status {exp['status']}: {exp['error']}", file=sys.stderr)
            return 1
        mid = ws.selected_model(eid)
        if a.test:
            rep = ws.evaluate_test(mid)
            k = "macro_f1" if task == "classification" else "rmse"
            print(f"[test] {mid}: {k}={rep[k]:.4f} (single use of the test set)")
        if a.export:
            ws.export(mid, a.export)
            print(f"[export] {a.export}")
        print(f"[ok] workspace={ws.root} experiment={eid} model={mid}")
        return 0
    except (DatasetError, WorkspaceError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    finally:
        ws.close()


def _parse_answers(pairs):
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        out[k.strip()] = v.strip()
    return out


def cmd_auto(a) -> int:
    """Let Epoch decide: description + dataset -> advice -> AutoML run -> test -> report/export."""
    from .advisor import apply_answers, describe_advice
    ws = Workspace(a.workspace or tempfile.mkdtemp(prefix="epoch_ws_"))
    try:
        did = ws.import_dataset(a.csv)
        hints = {}
        if a.target:
            hints["target"] = a.target
        if a.task != "auto":
            hints["task_type"] = a.task
        if a.minutes:
            hints["budget_minutes"] = a.minutes
        advice = ws.advise(did, a.describe or "", hints or None)
        answers = _parse_answers(a.answer)
        if answers:
            advice = apply_answers(advice, answers)
        print(describe_advice(advice))
        if advice.questions or advice.task_type == "unknown" or not advice.target:
            print("\nEpoch needs your answer before it can train. Re-run with --answer id=value, for example:", file=sys.stderr)
            for q in advice.questions:
                print(f"  --answer {q['id']}=<one of: {', '.join(map(str, q.get('options', []))[:200]) if False else ', '.join(str(o) for o in q.get('options', [])[:8])}>   ({q['text']})", file=sys.stderr)
            return 3
        over = {}
        if a.budget:
            over["budget_mode"] = a.budget
        if a.seconds:
            over["budget_seconds"] = a.seconds
        eid = ws.create_auto_experiment(did, advice, seed=a.seed, **over)
        ws.run_blocking(eid, lambda e: (lambda s: print(s) if s else None)(_fmt_event(e)))
        exp = ws.experiment(eid)
        if exp["status"] != "finished":
            print(f"Run ended with status {exp['status']}: {exp['error']}", file=sys.stderr)
            return 1
        mid = ws.selected_model(eid)
        if a.test:
            rep = ws.evaluate_test(mid)
            keys = ("macro_f1", "balanced_accuracy", "accuracy") if advice.task_type == "classification" else ("rmse", "mae", "r2")
            print("[test] locked test set (used once): " + ", ".join(f"{k}={rep[k]:.4f}" for k in keys if k in rep))
        if a.export:
            ws.export(mid, a.export)
            print(f"[export] {a.export} (includes report.html)")
        print(f"[ok] workspace={ws.root} experiment={eid} model={mid}")
        return 0
    except (DatasetError, WorkspaceError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    finally:
        ws.close()


def cmd_profile(a) -> int:
    ws = Workspace(a.workspace or tempfile.mkdtemp(prefix="et_ws_"))
    try:
        ds = ws.dataset(ws.import_dataset(a.csv))
        p = ds["profile"]
        print(f"{p['n_rows']} rows x {p['n_cols']} cols, duplicates={p['n_duplicate_rows']}")
        for c in p["columns"]:
            print(f"  {c['name']:<24} {c['kind']:<12} missing={c['pct_missing']}% unique={c['n_unique']}")
        print("target suggestions:", json.dumps(p["target_suggestions"]))
        return 0
    except DatasetError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    finally:
        ws.close()


def selftest() -> int:
    """Quick end-to-end check on generated data; used for smoke-testing a built app."""
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(0)
    n = 300
    df = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n), "c": rng.choice(["x", "y", "z"], n)})
    df["label"] = np.where(df.a + 0.5 * df.b + rng.randn(n) * 0.3 > 0, "pos", "neg")
    tmp = Path(tempfile.mkdtemp(prefix="et_selftest_"))
    csv = tmp / "selftest.csv"
    df.to_csv(csv, index=False)
    ws = Workspace(tmp / "ws")
    did = ws.import_dataset(csv)
    eid = ws.create_experiment(did, target="label", task_type="classification", mode="automl", budget_seconds=25, max_trials=2, n_jobs=2,
                               models=["linear", "tree", "lightgbm"])
    ws.run_blocking(eid)
    e = ws.experiment(eid)
    ok = e["status"] == "finished"
    if ok:
        mid = ws.selected_model(eid)
        ws.evaluate_test(mid)
        ws.export(mid, tmp / "bundle")
    ws.close()
    print("selftest", "OK" if ok else f"FAILED: {e['error']}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="epoch")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train", help="import a CSV, train candidates, evaluate, optionally export")
    t.add_argument("csv"); t.add_argument("--target", required=True)
    t.add_argument("--task", choices=["auto", "classification", "regression"], default="auto")
    t.add_argument("--workspace"); t.add_argument("--export"); t.add_argument("--seed", type=int, default=42)
    t.add_argument("--exclude", nargs="*", default=[]); t.add_argument("--dedupe", action="store_true")
    t.add_argument("--no-test", dest="test", action="store_false", help="skip the single test-set evaluation")
    u = sub.add_parser("auto", help="let Epoch decide: describe your data (or not) and it chooses the setup, trains, validates and reports")
    u.add_argument("csv"); u.add_argument("--describe", help="plain-language description of the data and goal (optional)")
    u.add_argument("--target"); u.add_argument("--task", choices=["auto", "classification", "regression"], default="auto")
    u.add_argument("--budget", choices=["express", "balanced", "deep"]); u.add_argument("--minutes", type=float); u.add_argument("--seconds", type=float)
    u.add_argument("--answer", nargs="*", help="answers to Epoch's questions, as id=value")
    u.add_argument("--workspace"); u.add_argument("--export"); u.add_argument("--seed", type=int, default=42)
    u.add_argument("--no-test", dest="test", action="store_false")
    p = sub.add_parser("profile", help="import and profile a CSV")
    p.add_argument("csv"); p.add_argument("--workspace")
    sub.add_parser("gui", help="launch the desktop application")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "train":
        return cmd_train(a)
    if a.cmd == "auto":
        return cmd_auto(a)
    if a.cmd == "profile":
        return cmd_profile(a)
    from .gui.app import run
    return run()
