---
title: Model export
order: 5
status: partial
statusnote: Export implemented in the engine.
desc: What is in an export bundle.
---
## Bundle contents
- `model.joblib`: the fitted pipeline (preprocessing plus estimator)
- `schema.json`: expected input columns and types
- `labels.json`: class label mapping (classification)
- `config.json`: task, split, seed and candidate configuration
- `metrics.json`: validation and test evaluation report
- `requirements.txt`: library versions used for training
- `manifest.json`: SHA-256 of every file
- `predict_example.py`: loads and uses the bundle

## Safety
> The model file is a joblib/pickle serialisation. Loading it can execute arbitrary code. Only load bundles you created or obtained from a source you trust, and check the checksum in `manifest.json` first.