---
title: Training workflow
order: 4
status: partial
statusnote: Tabular classification and regression only.
desc: Splits, candidates, selection and evaluation.
---
## Splits
The dataset is split into train, validation and test sets using a fixed seed. Classification uses stratified splitting. The split indices are stored with the experiment.

## Candidates
A baseline model is always trained first. Further candidates depend on the task: linear models and tree-based ensembles. The list of candidates actually tried is stored with the experiment.

## Selection
Candidates are compared on validation metrics only. Classification ranks by macro-F1; regression ranks by RMSE.

## Test set
The selected model is evaluated once on the test set. The application records this and warns if you request it again.

## Not implemented yet
- Hyperparameter search
- Cross-validation, group and temporal splits
- Pause and resume
- GPU training