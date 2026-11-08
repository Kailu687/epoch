---
title: Epoch — Handover
subtitle: State of the project at the end of session 3
version: 0.1.0.dev0
date: 2026-10-20
status: Handover
---

# 1. Summary

| Area | State |
|---|---|
| Technical design PDF | **Complete** (27 pages) |
| Website | **Complete (16 pages); not deployed** |
| Release manifest and downloads page | **Complete**; one verified development-channel artifact |
| Tabular training engine | **Complete for v0.1 scope** |
| Desktop front end | **Complete for v0.1 scope**; not hand-clicked |
| Windows packaging | **Partially complete**: unsigned zip, no installer |
| Other domains, HPO, CV, GPU, plugins | **Planned only** |

# Rename to Epoch (session 4)

- The product is now **Epoch**. All user-facing text, the website, docs, PDFs, window title, launchers (`Epoch.exe`, `EpochCLI.exe`), zip name and the workspace database name (`epoch.sqlite`; old `engine_trainer.sqlite` workspaces still open) were renamed. The internal Python package is still `engine_trainer`.
- Zip rebuilt as `Epoch-0.1.0.dev0-win-x64.zip` (238,311,200 bytes); frozen `EpochCLI.exe --selftest` OK; GUI launcher alive after 10 s; 31 app tests, 8 site tests, link check all pass.
- **Logo: not applied.** The owner said it was uploaded under `website/src/assets`, but no logo file exists there or anywhere in the project (only `favicon.svg`, the placeholder mark made in session 1). `website/build.py` now uses `src/assets/logo.svg|png|webp` automatically if present. The desktop app and favicon still use the placeholder.

# 2. What was designed

`docs/Epoch_Technical_Design.pdf` (source `.md`) covers requirements, user journeys, architecture and data flow, the universal provider contracts, dataset intelligence, AutoML, evaluation integrity, hardware, desktop design, website and release infrastructure, security, persistence, stack, testing, roadmap, cost, limitations, glossary and references.

# 3. What was implemented

## 3.1 Created files
- `docs/` design (md, pdf); `tools/md_to_pdf.py`, `tools/build_release.py`
- `website/` generator, link checker, tests, 4 page fragments (home, features, how it works, about), 9 docs, CSS, favicon, 5 screenshots, roadmap data; generated `dist/`
- `releases/releases.json` and `releases/artifacts/Epoch-0.1.0.dev0-win-x64.zip`
- `app/engine_trainer/` engine, CLI, GUI; `app/tests/`; PyInstaller spec and entry point; `requirements.txt`
- `README.md`, `PROJECT_STATUS.md/.pdf`, `NEXT_SESSION_PROMPT.md`, `HANDOVER.md/.pdf`

## 3.2 Changes in this session
- Website: Features page with capability status and real screenshots, home hero uses a real screenshot, Features added to navigation, docs updated for the two launchers and demo dataset, CSS fix for narrow screens
- Front end: planned features shown as labelled lines instead of look-alike controls, results bar chart against the baseline, synthetic demo dataset button, export defaults to the workspace folder and reports the true file count, narrower spin boxes
- Packaging: windowed GUI launcher plus a console CLI launcher; build script now also checks that the GUI launcher survives start-up

## 3.3 Application functionality
Import (background thread), profiling, task inference, validation and leakage warnings, seeded stratified split, train-only preprocessing, four candidates in a worker process with live progress and resource readout, cancel, validation-based selection, single-use test evaluation, SQLite history and registry, export bundle with checksum manifest and an inference loader that needs an explicit trust flag.

# 4. Testing

| Check | Result |
|---|---|
| `app/tests` (unit, end-to-end, GUI offscreen, demo dataset) | **31 passed** |
| `website/test_build.py` | **8 passed** |
| `website/check_site.py` | 16 pages, 363 internal refs, 1 download link: **OK** |
| Frozen `EpochCLI.exe --selftest` | **OK** |
| Frozen `Epoch.exe` start-up | Alive after 10 s |
| Zip integrity | `testzip` clean; served copy hash equals manifest |
| Mobile overflow (375 px), 8 pages | 1 px overflow on How it works, fixed with CSS; re-measured: 0 px on all 8 |

## 4.1 Not tested
- Clean Windows machine, Windows 10, macOS, Linux
- Manual click-through of the GUI, high DPI, small windows
- Website accessibility tooling and performance audit
- Large datasets; any GPU path (none exists)

# 5. Builds

| Build | Status |
|---|---|
| Website `dist/` | Built and link-checked; **not deployed** |
| Windows zip | Built; 238,311,200 bytes; SHA-256 `36df3bca73d855908d1f1b3702293ec75069987e88348465724ee7976419ba1e`; unsigned |
| Installer (MSI, Inno, MSIX) | **Not built** |
| macOS and Linux | **Not attempted** |

# 6. Available downloads

One: `releases/artifacts/Epoch-0.1.0.dev0-win-x64.zip` (development channel, manifest version `0.1.0-dev.0`). `website/build.py` copies it to `website/dist/downloads/0.1.0-dev.0/` after verifying size and hash. There is no stable release, and no remote URL has been verified.

# 7. Known bugs and limitations

- Unsigned; tested only on the build machine
- Windowed launcher hides errors (use the CLI launcher)
- Pickle-based export (trust required)
- Heuristic leakage probe; no refit on train+val
- `evaluate_test` and `export` re-read the CSV on the calling thread
- numpy 2.5 DeprecationWarning from joblib
- No git, no CI

# 8. Remaining architectural decisions

- Safer export format (ONNX or skops)
- Plugin trust and sandboxing
- Installer technology and signing
- Where to host large artifacts (the zip is 238 MB)

# 9. Completed, partial, planned

| Completed | Partial | Planned |
|---|---|---|
| Design PDF | Packaging (zip only) | Group/temporal splits, CV |
| Website (16 pages) and manifest gating | Docs (labelled partial) | Hyperparameter search |
| Tabular engine, CLI, tests | | Excel/Parquet import |
| Desktop front end (12 screens) | | GPU, pause/resume |
| Export bundle and trust-flag loader | | Plugin API |
| | | Image, audio, text, time-series, neural |
| | | Installer, signing, CI, deployment |

# 10. Exact next steps

1. Install git; commit.
2. Test the zip on a clean Windows machine.
3. Deploy `website/dist`; host the zip as a release asset and switch the manifest to a verified URL.
4. Implement group and temporal splits and cross-validation with tests.
5. Safer export format.
6. Signing, installer, CI.

Resume point: step 1–2. Reproduction commands are in `NEXT_SESSION_PROMPT.md`.
