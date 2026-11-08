"""Standalone inference helper. Shipped inside every export bundle: depends only on pandas, numpy, joblib, scikit-learn.

SECURITY: model.joblib is a pickle-based file. Loading it can execute arbitrary code. The manifest check below
detects accidental corruption or modification of this bundle's files, but it does NOT prove the bundle is
trustworthy: a malicious bundle can ship a matching manifest. Only load bundles you created or received from a source you trust.
"""
import hashlib
import json
import sys
from pathlib import Path


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_bundle(bundle_dir):
    """Return a list of problems (empty if every file matches manifest.json)."""
    d = Path(bundle_dir)
    mf = d / "manifest.json"
    if not mf.is_file():
        return ["manifest.json missing"]
    problems = []
    for name, digest in json.loads(mf.read_text(encoding="utf-8"))["files"].items():
        p = d / name
        if not p.is_file():
            problems.append(f"missing file: {name}")
        elif _sha256(p) != digest:
            problems.append(f"checksum mismatch: {name}")
    return problems


def prepare_input(df, schema):
    """Validate columns against schema.json and normalise dtypes exactly as during training."""
    import pandas as pd

    missing = [c["name"] for c in schema["columns"] if c["name"] not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    out = pd.DataFrame(index=df.index)
    for c in schema["columns"]:
        s = df[c["name"]]
        if c["kind"] == "numeric":
            out[c["name"]] = pd.to_numeric(s, errors="coerce").astype("float64")
        else:
            out[c["name"]] = s.astype("object").where(s.isna(), s.astype(str))
    return out


def load_bundle(bundle_dir, *, i_trust_this_bundle=False):
    """Load (model, schema, labels). Refuses unless the caller explicitly acknowledges the trust requirement."""
    if not i_trust_this_bundle:
        raise PermissionError("Loading a model bundle executes pickle data. Pass i_trust_this_bundle=True only for bundles from a trusted source.")
    problems = verify_bundle(bundle_dir)
    if problems:
        raise ValueError("Bundle integrity check failed: " + "; ".join(problems))
    import joblib

    d = Path(bundle_dir)
    schema = json.loads((d / "schema.json").read_text(encoding="utf-8"))
    labels = json.loads((d / "labels.json").read_text(encoding="utf-8")) if (d / "labels.json").exists() else None
    return joblib.load(d / "model.joblib"), schema, labels


def predict(bundle_dir, df, *, i_trust_this_bundle=False):
    model, schema, labels = load_bundle(bundle_dir, i_trust_this_bundle=i_trust_this_bundle)
    X = prepare_input(df, schema)
    return model.predict(X)


if __name__ == "__main__":
    import pandas as pd

    if len(sys.argv) < 3:
        sys.exit("usage: python inference.py <bundle_dir> <input.csv>  (loads the model: trusted bundles only)")
    out = predict(sys.argv[1], pd.read_csv(sys.argv[2]), i_trust_this_bundle=True)
    print("\n".join(map(str, out)))
