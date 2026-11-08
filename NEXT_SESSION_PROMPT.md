# Prompt for the next Claude session

You are continuing **Epoch**, a local-first ML training platform, in `C:\Users\Kailash A\Desktop\Projects\ML Model Trainer` (Windows, PowerShell, Python 3.12; git is not installed). **Do not rebuild completed work.** Read `PROJECT_STATUS.md` and `README.md` first, then inspect files before changing them.

## What already exists (verified)
- `docs/Epoch_Technical_Design.{md,pdf}` — complete 27-page design.
- `website/` — static site generator + 16 pages (incl. Features with real screenshots); `releases/releases.json` manifest; download links only appear for verified artifacts. One real artifact: `releases/artifacts/Epoch-0.1.0.dev0-win-x64.zip` (development channel, unsigned; windowed `Epoch.exe` + console `EpochCLI.exe`).
- `app/engine_trainer/` — working tabular engine (CSV → profile → validate → split → train baseline/logreg-ridge/RF/HistGB in a worker process → validation selection → single-use test eval → SQLite → export bundle), CLI, and PySide6 GUI with 12 screens.
- Tests: `cd app && python -m pytest tests -q` (31 pass); `cd website && python -m pytest test_build.py -q` (8 pass).
- Packaging: `python tools/build_release.py` (PyInstaller; runs frozen selftest).

## Highest-priority tasks (in order)
1. Install git (`winget install Git.Git` or ask the user), `git init`, first commit (`.gitignore` already exists).
2. Test the zip on a clean Windows machine (ask the user) and fix what breaks; consider a signed installer later.
3. Deploy `website/dist` (ask the user for a host). Host the 238 MB zip as a release asset, then in `releases/releases.json` replace `"file"` with `"url"` + `"url_verified_at"` only after fetching it and matching the SHA-256.
4. Group-aware and temporal splits + cross-validation (design doc §8) with tests (groups must never cross splits).
5. Safer export format (ONNX or skops); decision open (design §18).
6. Hand-click the GUI (high DPI, small windows); audit website accessibility.
7. CI.

Done already: threaded CSV import, results chart, demo dataset, windowed + CLI launchers, Features page, screenshots (regenerate with `$env:ET_NATIVE_QT='1'; $env:ET_SHOTS='<dir>'; python -m pytest tests/test_gui.py tests/test_gui_sample.py` and copy into `website/src/assets/img`).

## Rules
- Never fabricate metrics, sizes, checksums, versions or test results. Mark unavailable values as unavailable.
- Do not mark a release downloadable unless the file exists and its SHA-256 matches; `website/build.py` enforces this.
- Never use the test split for selection; keep `Workspace.evaluate_test` single-use semantics.
- When running PowerShell, avoid `Set-Content -Encoding UTF8` on source files (adds a BOM that broke the site build once); use Python or the Write tool.
- Keep `PROJECT_STATUS.md` and the handover accurate; regenerate `HANDOVER.pdf` with `python tools/md_to_pdf.py HANDOVER.md HANDOVER.pdf` after meaningful changes.

## Reproduce the environment
```
cd app
python -m pip install -r requirements.txt
python -m pytest tests -q
cd ../website && python build.py && python check_site.py
```
Resume point: task 1–2 above (version control, then real-GUI visual review).
