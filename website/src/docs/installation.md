---
title: Installation
order: 1
status: partial
statusnote: Unsigned dev zip or source; no installer exists.
desc: Install the development zip or run from source.
---
> There is no installer and no stable release. Two options: the unsigned development zip (Option A) or running from source (Option B).

## Option A: development zip (Windows x64)
1. Download the zip from the Downloads page and compare its SHA-256 with the value shown there.
2. Extract it. Windows SmartScreen may warn because the build is not code-signed.
3. Run `Epoch\Epoch.exe` to open the app (no console window).
4. For the command line, or to check the install, run `Epoch\EpochCLI.exe --selftest`. It should print `selftest OK`.

## Option B: from source

### Requirements
- Windows 10/11 x64 (other platforms untested)
- Python 3.12
- About 1 GB free disk space for dependencies

### Steps
```
cd app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m engine_trainer
```

### Verify the install
```
python -m engine_trainer --selftest
python -m pytest tests
```