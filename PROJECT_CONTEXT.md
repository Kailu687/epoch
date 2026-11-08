# Epoch — Project Context (living document)

**Read this first. Update it at the end of every work session.** It is the single source of truth for what Epoch is, what exists, who is doing what, and what is next. If this file and other docs disagree, check the code, then fix the docs.

Last updated: 2026-11-08 (session 5: AutoML v2, Advisor, Improve, Report, Try-model built; v0.2.0b1 packaged and published)

---

## 1. What Epoch is

Epoch (formerly "Engine Trainer"; the Python package is still `engine_trainer`) is a **local-first AutoML workbench**. A user gives it a dataset and, optionally, the kind of data or a plain-language description. Epoch works out the problem, picks the methods, trains and tunes candidate models, validates them honestly, and hands back the best validated model, a report, and a way to try it. The user can later **improve** a model with more data and get **recommendations** for what to change or which data to add.

Owner: Kailash A. Product site goal: `kailashamarnath.me` (products page) and `kailashamarnath.me/epoch`.

### Product promise (keep it honest)
> Give Epoch your dataset. It analyses it, chooses an appropriate machine-learning approach, runs budgeted experiments, validates the results, and delivers the best validated model it finds within your constraints, with a report and recommendations to improve it.

It does **not** promise the best possible model for every dataset. Some data is too small, mislabeled, or needs domain knowledge. Epoch must say so instead of manufacturing confidence.

### Prime-time feature: "Let Epoch decide" (the Advisor / micro-AI)
The user is often not an ML person. They can describe the data in plain words ("customer records, I want to know who will leave") or say "I don't know". Epoch's **Advisor** reads the description plus the dataset profile and chooses: task, target, metric, split strategy (including groups and time), training budget, optimisation profile, and which columns to drop. It explains each choice in plain language and asks a question only when it is unsure. It is deterministic and works offline; an optional local LLM backend may be added later but is never required.

### Core principles
1. Optimise generalisation, not impressive numbers. Train score is never the headline.
2. Understand the data before training (profile, quality, leakage, groups, time).
3. Preprocessing is fit only inside training folds (sklearn Pipelines).
4. Validation selects; the test set is locked and used once per final candidate (use is logged).
5. Spend compute on promising candidates (successive halving / Optuna), within an explicit time budget.
6. Everything is reproducible (seeds, split hashes, data hash, versions, config).
7. Never fabricate metrics, sizes, hashes, versions or benchmark claims. Mark unavailable values as unavailable.
8. Local-first and private. No telemetry, no required cloud. Treat datasets and model files as untrusted input.

---

## 2. Scope by phase

| Phase | Scope | Status |
|---|---|---|
| 1 | Tabular classification and regression (CSV; Excel/Parquet next) | v0.2.0b1 shipped: AutoML, Advisor, Improve, Report, Try-model. Excel/Parquet, ONNX export, calibration pending |
| 2 | Time series and signal data; audio (DSP features, spectrogram CNN, speaker-independent splits) | Planned |
| 3 | Images, text | Planned |
| 4 | Multimodal, compression, monitoring/drift, remote workers | Planned |

A supported file format is not a supported task. Each modality/task has its own pipeline, splitting and metrics.

---

## 3. Current state (verify against code before relying on it)

### Built and tested (v1, 2026-10-19/20)
- `app/engine_trainer/`: CSV import (background thread), profiling, validation + leakage probe, stratified/random split, sklearn pipelines, candidates (baseline, logistic/ridge, random forest, HistGradientBoosting), worker-process training with cancel, CPU/RAM readout, validation selection, single-use test evaluation, SQLite store, export bundle (joblib + schema + metrics + checksum manifest + standalone inference helper), CLI, PySide6 GUI (12 screens), PyInstaller build (`Epoch.exe`, `EpochCLI.exe`).
- Tests: `cd app && python -m pytest tests -q` (31 pass at last run). Website tests: `cd website && python -m pytest test_build.py -q` (8 pass).
- Website generator in `website/` (16 pages), release manifest `releases/releases.json`, one unsigned dev zip.
- Docs: `docs/Epoch_Technical_Design.{md,pdf}` (27 pages, v1-era design), `PROJECT_STATUS.pdf`, `HANDOVER.pdf`.

### Built in session 5 (v0.2.0b1)
- `automl/`: model zoo (12 families, optional LightGBM/XGBoost/Optuna), screening -> Optuna TPE tuning with median pruning -> full-data CV with out-of-fold predictions -> Caruana ensemble (kept only if gain > margin) -> constraint-aware choice (profile) -> learning curve + recommendations. Splits: stratified/random/group/temporal; CV schemes match. Budgets: express 150 s, balanced 1500 s, deep 7200 s (soft; refit/curve may overrun).
- `advisor/` (built by thread `epoch-advisor`, 47 tests): deterministic offline Advisor. No accuracy figure is claimed for it.
- `improve/` + `Workspace.improve_model/compare_improvement`: old rows + new rows retrained; ORIGINAL test rows stay the locked test; part of the new rows is a second held-out test; compatibility/drift report; verdict adopt/keep_old/inconclusive.
- `report/html.py`, `Workspace.report_html`, `try_model`, `summary`; export bundle now includes `report.html`.
- GUI: Task ("Let Epoch decide"), Models, Effort, Monitor (search events), Results (CV, advice, report), Try model, Improve; pages scroll. CLI: `epoch auto`.
- Tests: 92+ (app) before the improve/GUI additions; see log for the final count.

### Known gaps vs the vision (after session 5)
No Excel/Parquet import, no calibration, no SHAP/permutation importance, no ONNX/safer export, no GPU use, no pause/resume, no drift monitoring after deployment, no CatBoost, no audio/image/text/time-series, no suggestions of specific external datasets (recommendations are generic: more rows / columns / classes), no windowed-GUI error log, unsigned build, not tested on a clean Windows machine.

### Environment (this machine)
Windows 11, Python 3.12.4, no git on the ML Model Trainer folder (the Portfolio folder is a git repo). Installed: numpy, pandas, scikit-learn, xgboost 3.2, scipy, joblib, psutil, PySide6 6.8, torch 2.14 CPU-only, librosa, pyarrow, openpyxl, pyinstaller. Added in session 5 (optional in code, graceful fallback): lightgbm 4.7, optuna 5.0. Not installed: catboost, shap, mlflow, flaml, autogluon, onnx, skops, polars, duckdb. No CUDA on this Python.

---

## 4. Target architecture (v2)

```
app/engine_trainer/
  io_csv.py profile.py validate.py splits.py pipeline.py evaluate.py store.py export.py inference.py   (v1, keep)
  advisor/        NEW  micro-AI: description + profile -> Advice (task, target, metric, split, budget, profile, drops, explanations)
  automl/         NEW  planner, model zoo, search (Optuna/successive halving), CV, ensembling, selection under constraints
  improve/        NEW  continue-training / refit with new data, drift/compat checks between old and new data, improvement recommendations
  report/         NEW  HTML + JSON evaluation report, model card, "try your model" prediction helpers
  runner.py jobs.py workspace.py cli.py gui/   (v1 orchestration, extended)
```

Contracts (keep stable; other threads depend on them):
- **Profile dict**: output of `profile.profile_dataset(df)` (keys `n_rows, n_cols, n_duplicate_rows, columns[{name, kind, dtype, n_missing, pct_missing, n_unique, example_values, notes}], target_suggestions`). The Advisor and planner read this.
- **Advice** (Advisor output, JSON-serialisable dataclass): `task_type` (`classification|regression|unknown`), `target` (+ ranked `target_candidates` with reasons), `data_kind` (`tabular` for now), `metric`, `split_strategy` (`stratified|random|group|temporal`), `group_column`, `time_column`, `budget_mode` (`express|balanced|deep`), `optimisation_profile` (`balanced|accuracy_first|speed_first|lightweight|interpretable`), `exclude_columns`, `warnings[]`, `explanations[]` (plain language, one per decision), `questions[]` (only when needed), `confidence` (`high|medium|low`).
- **Plan** (automl planner output): ordered candidate list, per-candidate search space, time budget split, CV scheme, metric, constraints.
- **Run summary** (`summary.json` per experiment) already exists; v2 adds `cv`, `search`, `ensemble`, `recommendations`.
- **Test-set rule**: `Workspace.evaluate_test` stays single-use unless `acknowledge_repeat=True`.

---

## 5. Work split (threads)

| Thread | Owner session | Scope | Files it may edit |
|---|---|---|---|
| Core (this session) | main | AutoML v2 engine, improve workflow, reports, GUI wiring, tests, this document | `app/engine_trainer/automl|improve|report|runner|workspace|cli|gui`, tests, docs |
| Advisor | `epoch-advisor` | micro-AI Advisor module | `app/engine_trainer/advisor/**`, `app/tests/test_advisor*.py` only |
| Website | `epoch-site` | kailashamarnath.me products page and `/epoch` | `C:\Users\Kailash A\Desktop\Projects\Portfolio` only (separate git repo; no push/deploy without the owner's OK) |

Rule: one writer per file set. Threads report back; core integrates. After each thread finishes, record it in the log below.

---

## 6. Backlog (priority order)

1. (DONE in v0.2.0b1) AutoML v2 core: model zoo (adds LightGBM/XGBoost/ExtraTrees/SVM/KNN/MLP/NB; CatBoost optional), Optuna search with pruning, repeated/grouped CV, budgets (express/balanced/deep), constraint-aware selection (profile), safe ensembling (weighted average + stacking with out-of-fold), calibration.
2. Advisor integration: "Let Epoch decide" in GUI and CLI (`epoch auto data.csv --describe "..."`).
3. Improve workflow: pick an existing model, add a new dataset, check schema compatibility and drift, refit/continue, compare old vs new on the same locked test, accept or reject.
4. Recommendations: diagnose (over/underfit, imbalance, label noise suspicion, small data, leakage, instability) and suggest actions plus what kinds of data to add (learning-curve based: "more rows help" vs "more features/classes help").
5. Report: HTML + JSON (metrics, CV spread, confusion matrix, calibration, importance, error analysis, limits, reproducibility); Try-your-model panel (CSV in, predictions out; single-row form).
6. Excel/Parquet/JSON import; group/temporal split UI.
7. Safer export (ONNX/skops), MLflow-compatible tracking export.
8. Phase 2+: time series/signals, audio (reuse SER code from `Signal Flow`: actor-independent splits), images, text.
9. Packaging: rebuild zip, sign, installer, clean-machine test. Deployment of site (owner's call).
10. Benchmark suite (clean, imbalanced, small, grouped, leaky, missing data) vs reference AutoML under equal budgets. Only publish numbers that were actually measured.

---

## 7. Decisions (append, do not rewrite)

- 2026-10-20: Product renamed Engine Trainer -> Epoch. Package name stays `engine_trainer`.
- 2026-10-19: PySide6 UI, training in spawned worker processes, SQLite metadata, static website via Python generator.
- 2026-11-08: Vision expanded (ChatGPT-sourced AutoML spec used as inspiration, not as a spec to copy). Advisor ("Let Epoch decide") is the headline feature. External AutoML frameworks (AutoGluon, FLAML) are NOT bundled: they are heavy and hurt the packaged build; Epoch implements its own planner/search on scikit-learn + LightGBM/XGBoost/Optuna, with optional extras detected at runtime.
- 2026-11-08: New dependencies must be optional at import time (graceful fallback) so the frozen app and tests keep working without them.

---

## 8. Change log (newest first; add an entry every session)

- 2026-11-08 (later) — Built AutoML v2, integrated Advisor (thread `epoch-advisor`), Improve workflow, HTML report, Try-model, GUI v2, `epoch auto`. Version 0.2.0b1. PyInstaller build failed twice with `PermissionError` on the fresh exe (likely antivirus lock) in `build/` on the Desktop; fixed by building in `%TEMP%\epoch_build` via `EPOCH_BUILD_DIR`. Source pushed to public repo github.com/kailu687/epoch (owner chose public). Website thread `epoch-site` built /epoch pages on branch `epoch-product-page` of the Portfolio repo; downloads page reads GitHub Releases of kailu687/epoch.

- 2026-11-08 — Context created. Threads planned: Advisor, Website. Core AutoML v2 started.
