---
title: Quick start
order: 2
status: partial
statusnote: GUI and command line run from source.
desc: Train your first model.
---
1. Start the app (`Epoch.exe` in the zip, or `python -m engine_trainer` from source), or use the command line below.
2. Choose a workspace folder.
3. Import a CSV file, or press "Create a synthetic demo dataset" on the Import screen to try the workflow with generated data. Pick the target column and confirm the task type.
4. Start training and wait for the candidates to finish.
5. Review validation metrics, then evaluate the selected model on the test set.
6. Export the bundle.

## Command line equivalent
```
python -m engine_trainer train data.csv --target label --workspace my_ws --export out_dir
```
In the zip, use `EpochCLI.exe train ...` instead of `python -m engine_trainer train ...`.