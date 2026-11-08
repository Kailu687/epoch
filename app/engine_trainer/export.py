"""Export bundle writer."""
from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path

from . import __version__
from . import inference as _inference
from .io_csv import sha256_file

EXAMPLE = '''"""Example: load the exported model and predict. Run from this folder.

WARNING: model.joblib is a pickle-based file. Only load bundles from a source you trust.
"""
import pandas as pd
from inference import load_bundle, prepare_input, verify_bundle

print("integrity problems:", verify_bundle("."))
model, schema, labels = load_bundle(".", i_trust_this_bundle=True)  # explicit trust acknowledgement
print("expected columns:", [c["name"] for c in schema["columns"]])

rows = pd.DataFrame([{ROW}])   # replace with your own data
print("prediction:", model.predict(prepare_input(rows, schema)))
'''


def _inference_source() -> Path:
    """Locate inference.py as a real file; in a frozen build it is shipped as a data file (see engine_trainer.spec)."""
    import sys
    for p in (Path(_inference.__file__).with_suffix(".py"), Path(getattr(sys, "_MEIPASS", ".")) / "engine_trainer" / "inference.py"):
        if p.is_file():
            return p
    raise FileNotFoundError("inference.py not found; cannot build a standalone bundle")


def env_versions() -> dict:
    import joblib, numpy, pandas, sklearn, scipy

    env = {"python": platform.python_version(), "platform": platform.platform(), "scikit-learn": sklearn.__version__, "pandas": pandas.__version__,
           "numpy": numpy.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__, "engine_trainer": __version__}
    for mod in ("lightgbm", "xgboost", "optuna"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    return env


def write_bundle(dest: Path, *, model_path: Path, schema_cols: list, classes, config: dict, report: dict, env: dict, example_row: dict,
                 extra_files: dict | None = None) -> dict:
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Export folder is not empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, dest / "model.joblib")
    (dest / "schema.json").write_text(json.dumps({
        "columns": schema_cols,
        "notes": ["numeric columns are converted to float (non-numeric values become missing)",
                  "categorical columns are converted to text; unseen categories are tolerated",
                  "missing values are imputed by the pipeline using training-set statistics"]}, indent=2), encoding="utf-8")
    if classes is not None:
        (dest / "labels.json").write_text(json.dumps({"classes": classes, "note": "predict() returns these original labels"}, indent=2), encoding="utf-8")
    (dest / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    (dest / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pins = [f"scikit-learn=={env['scikit-learn']}", f"pandas=={env['pandas']}", f"numpy=={env['numpy']}", f"scipy=={env['scipy']}", f"joblib=={env['joblib']}"]
    pins += [f"{k}=={env[k]}" for k in ("lightgbm", "xgboost") if k in env]  # only if installed; models built with them need them to load
    (dest / "requirements.txt").write_text("\n".join(pins) + "\n", encoding="utf-8")
    shutil.copy2(_inference_source(), dest / "inference.py")
    for name, content in (extra_files or {}).items():
        (dest / name).write_text(content, encoding="utf-8")
    row = ", ".join(f"{k!r}: {v!r}" for k, v in example_row.items())
    (dest / "predict_example.py").write_text(EXAMPLE.replace("{ROW}", "{" + row + "}"), encoding="utf-8")
    (dest / "README.md").write_text(
        "# Epoch export\n\nFiles: model.joblib, schema.json, labels.json (classification), config.json, metrics.json, "
        "requirements.txt, inference.py, predict_example.py, manifest.json.\n\n"
        "**Security:** model.joblib is pickle-based; loading can run arbitrary code. Load only bundles you trust. "
        "manifest.json detects modification of files, not malicious origin.\n\n"
        f"Created with Epoch {env['engine_trainer']}. Use the library versions in requirements.txt; other versions may not load the model.\n",
        encoding="utf-8")
    files = {p.name: sha256_file(p) for p in sorted(dest.iterdir()) if p.name != "manifest.json"}
    manifest = {"created_with": f"engine_trainer {env['engine_trainer']}", "files": files}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    problems = _inference.verify_bundle(dest)
    if problems:
        raise RuntimeError(f"Export verification failed: {problems}")
    return manifest
