# Epoch

**Give Epoch a dataset. It works out the problem, picks the methods, trains and tunes models, validates them honestly, and hands back the best validated model with a report.**

Epoch is a local-first AutoML workbench for Windows. Your data never leaves your computer. You do not need to know machine learning: describe your data in plain words (or say "I don't know") and Epoch chooses the task, target, score, split, effort and models, and explains each choice.

Status: **beta (0.2.0b1)**, tabular data (CSV) only. Images, audio, text and time-series models are planned, not available.

## Download

Get the latest Windows build from the [Releases page](https://github.com/kailu687/epoch/releases). Extract the zip and run `Epoch\Epoch.exe`. The build is **not code-signed**, so Windows SmartScreen may warn. Check the SHA-256 listed in the release notes against the file you downloaded:

```powershell
Get-FileHash .\Epoch-0.2.0b1-win-x64.zip -Algorithm SHA256
```

## What it does today

- **Let Epoch decide**: a plain-language description plus the data, and Epoch picks the setup and asks only when it is unsure.
- **AutoML search**: baseline, linear, trees, random forest, extra trees, gradient boosting, LightGBM, XGBoost (regression), SVM, k-NN, small neural net, naive Bayes; screening, Optuna tuning with pruning, cross-validation, optional ensembling kept only when the gain is meaningful.
- **Honest evaluation**: stratified, group-aware and temporal splits; preprocessing fitted inside folds; a locked test set that is used once and logged; leakage warnings.
- **Report and advice**: HTML report, per-class results, learning curve, and recommendations (more rows vs more columns, weak classes, instability, possible leakage).
- **Try your model**: predict a CSV or one example without code.
- **Improve a model**: add a new dataset; Epoch checks compatibility and drift, retrains on old + new data, and compares old vs new on the *same* locked test rows plus a held-out slice of the new data.
- **Export**: model, schema, labels, config, metrics, report, requirements, inference code, checksum manifest. Model files are pickle-based: load only bundles you trust.

Not available yet: GPU use, pause/resume, Excel/Parquet import, non-pickle export formats, images/audio/text/time-series, drift monitoring after deployment.

## Run from source

```powershell
cd app
python -m pip install -r requirements.txt
python -m engine_trainer                       # GUI
python -m engine_trainer auto data.csv --describe "Customer records, who will leave?" --budget express --export out
python -m engine_trainer --selftest
python -m pytest tests -q
```

## Build the Windows app

```powershell
python tools/build_release.py        # PyInstaller one-folder app, frozen self-test, zip and SHA-256
```

## Project layout

`app/engine_trainer/` engine (`advisor/`, `automl/`, `improve/`, `report/`, `gui/`), `app/tests/`, `website/` static site generator, `releases/releases.json` release manifest, `docs/` design document, `PROJECT_CONTEXT.md` living project context.

## License

No license has been chosen yet; all rights are reserved by the author.
