"""Budgeted model search: screen the zoo, tune the promising families, then try a safe ensemble.

Every score comes from cross-validation inside the development data (train + validation actors/rows). The locked test set
is never seen here. All preprocessing is fit inside each training fold.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit

from .. import pipeline as pl
from . import metrics as mx
from . import zoo

BUDGET_SECONDS = {"express": 150, "balanced": 1500, "deep": 7200}
FINALISTS = {"express": 2, "balanced": 4, "deep": 6}
MAX_TRIALS = {"express": 6, "balanced": 25, "deep": 80}
SCREEN_ROWS = 20000


class Cancelled(Exception):
    pass


class BudgetExceeded(Exception):
    pass


@dataclass
class SearchConfig:
    task: str
    metric: str
    kinds: dict
    classes: list | None
    seed: int = 42
    n_jobs: int = 1
    budget_seconds: float = 150
    mode: str = "express"
    profile: str = "balanced"
    cv_scheme: str = "stratified"  # stratified | kfold | group | temporal
    n_splits: int = 5
    max_trials: int | None = None
    n_finalists: int | None = None
    only: list | None = None  # restrict to these model ids


@dataclass
class CandResult:
    model_id: str
    params: dict
    cv_mean: float  # higher is better (see metrics.score)
    cv_std: float
    fold_scores: list
    default_mean: float | None = None
    tuned: bool = False
    n_trials: int = 0
    fit_seconds: float = 0.0
    oof_pred: np.ndarray | None = None
    oof_proba: np.ndarray | None = None
    covered: np.ndarray | None = None
    status: str = "ok"
    error: str | None = None


# ---------------------------------------------------------------- folds
def make_folds(y, task: str, scheme: str, n_splits: int, seed: int, groups=None):
    y = np.asarray(y)
    idx = np.arange(len(y))
    if scheme == "temporal":
        return list(TimeSeriesSplit(n_splits=max(2, n_splits)).split(idx))
    if scheme == "group" and groups is not None:
        ng = len(np.unique(groups))
        k = max(2, min(n_splits, ng))
        try:
            if task == "classification":
                return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(idx, y, groups))
        except ValueError:
            pass
        return list(GroupKFold(n_splits=k).split(idx, y, groups))
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        k = max(2, min(n_splits, int(counts.min())))
        if counts.min() >= 2:
            return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(idx, y))
    return list(KFold(n_splits=max(2, min(n_splits, len(y) // 2)), shuffle=True, random_state=seed).split(idx))


# ---------------------------------------------------------------- CV evaluation
class Clock:
    def __init__(self, budget: float, cancelled: Callable[[], bool]):
        self.t0 = time.perf_counter()
        self.budget = budget
        self.cancelled = cancelled

    def elapsed(self):
        return time.perf_counter() - self.t0

    def left(self):
        return self.budget - self.elapsed()

    def check(self, hard=True):
        if self.cancelled():
            raise Cancelled()
        if hard and self.left() <= 0:
            raise BudgetExceeded()


def cv_evaluate(model_id: str, params: dict, X, y, folds, cfg: SearchConfig, clock: Clock, want_oof=False, trial=None) -> CandResult:
    t0 = time.perf_counter()
    scores, n = [], len(y)
    oof_pred = oof_proba = covered = None
    if want_oof:
        covered = np.zeros(n, bool)
        oof_pred = np.empty(n, dtype=object if cfg.task == "classification" else float)
        if cfg.task == "classification":
            oof_proba = np.zeros((n, len(cfg.classes)))
    for i, (tr, va) in enumerate(folds):
        clock.check(hard=False)
        if clock.left() < -cfg.budget_seconds * 0.5:  # far past the budget: stop even mid-evaluation
            raise BudgetExceeded()
        est = zoo.build(model_id, params, cfg.task, cfg.seed, cfg.n_jobs)
        model = pl.full_pipeline(cfg.kinds, est)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X.iloc[tr], y[tr])
            pred = model.predict(X.iloc[va])
            proba = model.predict_proba(X.iloc[va]) if cfg.task == "classification" and hasattr(model, "predict_proba") else None
        if proba is not None and proba.shape[1] != len(cfg.classes):  # a fold missing a class: align columns
            full = np.zeros((len(va), len(cfg.classes)))
            for j, c in enumerate(model.classes_):
                full[:, cfg.classes.index(c)] = proba[:, j]
            proba = full
        scores.append(mx.score(cfg.metric, y[va], pred, proba, cfg.classes) if (proba is not None or cfg.metric not in mx.NEEDS_PROBA) else float("nan"))
        if want_oof:
            covered[va] = True
            oof_pred[va] = pred
            if proba is not None:
                oof_proba[va] = proba
        if trial is not None:
            trial.report(float(np.nanmean(scores)), i)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
    sc = np.array(scores, float)
    return CandResult(model_id, params, float(np.nanmean(sc)), float(np.nanstd(sc)), sc.tolist(), fit_seconds=time.perf_counter() - t0,
                      oof_pred=oof_pred, oof_proba=oof_proba, covered=covered)


def _subsample(y, task, n, seed):
    if len(y) <= n:
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    if task == "classification":
        keep = []
        for c in np.unique(y):
            ii = np.flatnonzero(y == c)
            keep.append(rng.choice(ii, max(2, int(round(len(ii) * n / len(y)))), replace=False))
        return np.sort(np.concatenate(keep))
    return np.sort(rng.choice(len(y), n, replace=False))


# ---------------------------------------------------------------- the search
def run_search(X, y, groups, cfg: SearchConfig, clock: Clock, emit: Callable[[str, dict], None]) -> dict:
    """Returns {'results': [CandResult...], 'finalists': [...], 'ensemble': {...} | None, 'folds': [...], 'log': [...]}."""
    y = np.asarray(y)
    log: list[str] = []
    folds = make_folds(y, cfg.task, cfg.cv_scheme, cfg.n_splits, cfg.seed, groups)
    defs = zoo.available(cfg.task, len(y), cfg.profile)
    if cfg.only:
        defs = [d for d in defs if d.id in cfg.only or d.id == "baseline"]
    emit("search_plan", {"models": [d.id for d in defs], "folds": len(folds), "scheme": cfg.cv_scheme, "budget_seconds": cfg.budget_seconds})

    # ---- stage 1: screen every family with default settings (cheap folds / subsample on big data)
    sub = _subsample(y, cfg.task, SCREEN_ROWS, cfg.seed)
    Xs, ys = X.iloc[sub].reset_index(drop=True), y[sub]
    gs = None if groups is None else np.asarray(groups)[sub]
    sfolds = folds if len(sub) == len(y) else make_folds(ys, cfg.task, cfg.cv_scheme, min(3, cfg.n_splits), cfg.seed, gs)
    screened: dict[str, CandResult] = {}
    screen_deadline = cfg.budget_seconds * 0.25
    for d in defs:
        try:
            clock.check(hard=False)
            if clock.elapsed() > screen_deadline and screened:
                if not any(x.startswith("Screening stopped") for x in log):
                    log.append("Screening stopped early to stay within the time budget; slower model families were not tried.")
                emit("candidate_skipped", {"id": d.id, "reason": "time budget"})
                continue
            emit("candidate_start", {"id": d.id, "stage": "screening"})
            r = cv_evaluate(d.id, d.defaults, Xs, ys, sfolds, cfg, clock)
            r.default_mean = r.cv_mean
            screened[d.id] = r
            emit("screened", {"id": d.id, "cv_mean": r.cv_mean, "cv_std": r.cv_std, "seconds": round(r.fit_seconds, 2)})
        except Cancelled:
            raise
        except BudgetExceeded:
            break
        except Exception as e:  # a single failing family must not stop the run
            screened[d.id] = CandResult(d.id, d.defaults, float("-inf"), 0.0, [], status="failed", error=f"{type(e).__name__}: {e}")
            emit("candidate_failed", {"id": d.id, "error": f"{type(e).__name__}: {e}"})
    ok = [r for r in screened.values() if r.status == "ok" and r.model_id != "baseline"]
    if not ok:
        return {"results": list(screened.values()), "finalists": [], "ensemble": None, "folds": folds, "log": log}
    ok.sort(key=lambda r: -r.cv_mean)
    k = cfg.n_finalists or FINALISTS.get(cfg.mode, 3)
    finalists_ids = [r.model_id for r in ok[:k]]

    # ---- stage 2: tune each finalist with a share of the remaining time
    tuned: dict[str, CandResult] = {}
    max_trials = cfg.max_trials or MAX_TRIALS.get(cfg.mode, 10)
    for i, mid in enumerate(finalists_ids):
        d = zoo.ZOO[mid]
        share = max(0.0, clock.left() * 0.55) / (len(finalists_ids) - i)
        best_params, best_val, n_trials = dict(d.defaults), screened[mid].cv_mean, 0
        tuned_flag = False
        if d.space is not None and share > 2:
            emit("candidate_start", {"id": mid, "stage": "tuning"})
            deadline = time.perf_counter() + share
            try:
                best_params, best_val, n_trials = _tune(d, Xs, ys, sfolds, cfg, clock, deadline, max_trials, emit, gs)
                tuned_flag = n_trials > 0
            except Cancelled:
                raise
            except BudgetExceeded:
                pass
            except Exception as e:
                log.append(f"Tuning {mid} failed ({type(e).__name__}: {e}); default settings kept.")
        tuned[mid] = CandResult(mid, best_params, best_val, 0.0, [], default_mean=screened[mid].cv_mean, tuned=tuned_flag, n_trials=n_trials)

    # ---- stage 3: full-data CV with OOF predictions for the finalists (needed for honest ensembling and the report)
    finals: list[CandResult] = []
    for mid in finalists_ids:
        try:
            clock.check(hard=False)
            emit("candidate_start", {"id": mid, "stage": "validating"})
            r = cv_evaluate(mid, tuned[mid].params, X, y, folds, cfg, clock, want_oof=True)
            if tuned[mid].tuned and len(sub) != len(y):
                # guard against tuning on a subsample hurting the full data: compare with defaults on the same folds
                rd = cv_evaluate(mid, zoo.ZOO[mid].defaults, X, y, folds, cfg, clock, want_oof=True)
                if rd.cv_mean >= r.cv_mean:
                    r, tuned[mid].tuned = rd, False
                    r.params = dict(zoo.ZOO[mid].defaults)
            r.default_mean, r.tuned, r.n_trials = tuned[mid].default_mean, tuned[mid].tuned, tuned[mid].n_trials
            finals.append(r)
            emit("validated", {"id": mid, "cv_mean": r.cv_mean, "cv_std": r.cv_std})
        except Cancelled:
            raise
        except BudgetExceeded:
            log.append(f"Out of time before {mid} could be validated on the full data.")
            break
        except Exception as e:
            emit("candidate_failed", {"id": mid, "error": f"{type(e).__name__}: {e}"})
    finals.sort(key=lambda r: -r.cv_mean)

    # ---- stage 4: ensemble (only kept if the gain is meaningful)
    ens = None
    if len(finals) >= 2:
        try:
            ens = build_ensemble(finals, y, cfg)
        except Exception as e:
            log.append(f"Ensembling skipped ({type(e).__name__}: {e}).")
    if "baseline" in screened and screened["baseline"].status == "ok":
        try:
            b = cv_evaluate("baseline", {}, X, y, folds, cfg, Clock(1e9, clock.cancelled), want_oof=True)  # trivial cost: never skipped for time
            b.default_mean = b.cv_mean
            screened["baseline"] = b
        except Exception:
            pass
    return {"results": list(screened.values()), "finalists": finals, "ensemble": ens, "folds": folds, "log": log}


def _tune(d, Xs, ys, folds, cfg, clock, deadline, max_trials, emit, groups):
    """Optuna TPE with median pruning, or random search when Optuna is missing. Returns (best_params, best_score, n_trials)."""
    if zoo.HAS_OPTUNA:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed, n_startup_trials=min(5, max_trials)),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1))

        def objective(trial):
            if time.perf_counter() > deadline:
                trial.study.stop()
            clock.check(hard=False)
            p = d.space(zoo.TrialSampler(trial), cfg.task, len(ys))
            try:
                r = cv_evaluate(d.id, p, Xs, ys, folds, cfg, clock, trial=trial)
            except BudgetExceeded:  # out of time mid-trial: end the study quietly
                trial.study.stop()
                raise optuna.TrialPruned()
            emit("trial", {"id": d.id, "number": trial.number, "value": r.cv_mean})
            return r.cv_mean

        study.optimize(objective, n_trials=max_trials, timeout=max(1.0, deadline - time.perf_counter()), catch=(BudgetExceeded,))
        done = [t for t in study.trials if t.state.name == "COMPLETE"]
        if not done:
            return dict(d.defaults), float("-inf"), 0
        return d.space(zoo.TrialSampler(optuna.trial.FixedTrial(study.best_params)), cfg.task, len(ys)), study.best_value, len(done)
    sampler = zoo.RandomSampler(cfg.seed)
    best_p, best_v, n = dict(d.defaults), float("-inf"), 0
    for t in range(max_trials):
        if time.perf_counter() > deadline:
            break
        clock.check(hard=False)
        p = d.space(sampler, cfg.task, len(ys))
        r = cv_evaluate(d.id, p, Xs, ys, folds, cfg, clock)
        n += 1
        emit("trial", {"id": d.id, "number": t, "value": r.cv_mean})
        if r.cv_mean > best_v:
            best_p, best_v = p, r.cv_mean
    return best_p, best_v, n


# ---------------------------------------------------------------- ensembling
def _combine(finals, weights, cfg):
    """Weighted average of OOF probabilities (classification) or predictions (regression)."""
    if cfg.task == "classification":
        p = sum(w * f.oof_proba for f, w in zip(finals, weights)) / max(sum(weights), 1e-12)
        return np.asarray(cfg.classes)[p.argmax(1)], p
    return sum(w * f.oof_pred.astype(float) for f, w in zip(finals, weights)) / max(sum(weights), 1e-12), None


def build_ensemble(finals: list[CandResult], y, cfg: SearchConfig, rounds: int = 12) -> dict | None:
    """Greedy forward selection with replacement (Caruana et al.) on out-of-fold predictions."""
    y = np.asarray(y)
    cov = np.logical_and.reduce([f.covered for f in finals])
    if cov.sum() < 10 or (cfg.task == "classification" and any(f.oof_proba is None for f in finals)):
        return None
    chosen: list[int] = []
    best_score = -np.inf
    for _ in range(rounds):
        trial_scores = []
        for j in range(len(finals)):
            w = np.bincount(chosen + [j], minlength=len(finals)).astype(float)
            pred, proba = _combine(finals, w, cfg)
            trial_scores.append(mx.score(cfg.metric, y[cov], pred[cov], None if proba is None else proba[cov], cfg.classes))
        j = int(np.argmax(trial_scores))
        if trial_scores[j] <= best_score + 1e-9 and chosen:
            break
        best_score = trial_scores[j]
        chosen.append(j)
    w = np.bincount(chosen, minlength=len(finals)).astype(float)
    pred, proba = _combine(finals, w, cfg)
    single = finals[0]
    sp, spr = _combine(finals, np.eye(len(finals))[0], cfg)  # best single model scored the same pooled way
    single_pooled = mx.score(cfg.metric, y[cov], sp[cov], None if spr is None else spr[cov], cfg.classes)
    gain = best_score - single_pooled
    margin = max(0.002, 0.25 * single.cv_std)
    members = [(finals[i].model_id, float(w[i] / w.sum())) for i in range(len(finals)) if w[i] > 0]
    return {"members": members, "weights": (w / w.sum()).tolist(), "score": float(best_score), "best_single": float(single_pooled),
            "gain": float(gain), "margin": float(margin), "accepted": bool(gain > margin and len(members) >= 2),
            "oof_pred": pred, "oof_proba": proba, "covered": cov}
