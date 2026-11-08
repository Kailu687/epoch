import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine_trainer.workspace import Workspace  # noqa: E402


def make_cls(n=500, seed=0, noise=0.3, planted_leak=False, id_col=False):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n), "cat": rng.choice(["x", "y", "z"], n)})
    score = df.a + 0.5 * df.b + (df.cat == "x") * 0.8 + rng.randn(n) * noise
    df["label"] = np.where(score > 0, "pos", "neg")
    df.loc[rng.choice(n, 20, replace=False), "a"] = np.nan  # some missing values
    if planted_leak:
        df["post_outcome"] = (df.label == "pos").astype(int) + rng.randn(n) * 0.01
    if id_col:
        df.insert(0, "customer_id", [f"C{i:05d}" for i in range(n)])
    return df


def make_noise(n=500, seed=1):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n)})
    df["label"] = rng.choice(["p", "q"], n)
    return df


def make_reg(n=500, seed=2):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n), "g": rng.choice(["u", "v"], n)})
    df["y"] = 3 * df.x1 - 2 * df.x2 + (df.g == "u") * 1.5 + rng.randn(n) * 0.3
    return df


@pytest.fixture
def ws(tmp_path):
    w = Workspace(tmp_path / "ws")
    yield w
    w.close()


@pytest.fixture
def csv_of(tmp_path):
    def _w(df, name="d.csv"):
        p = tmp_path / name
        df.to_csv(p, index=False)
        return p
    return _w
