"""Deterministic train/validation/test splitting returning index sets."""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

MIN_PER_CLASS = 3


class SplitError(Exception):
    pass


def make_split(y, task_type: str, seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15) -> dict:
    """Return {'train','val','test'} integer position arrays (into the cleaned frame).

    Classification is stratified. For group-aware and temporal protocols use `make_split_strategy`.
    """
    y = np.asarray(y)
    n = len(y)
    idx = np.arange(n)
    strat = None
    if task_type == "classification":
        _, counts = np.unique(y, return_counts=True)
        if counts.min() < MIN_PER_CLASS:
            raise SplitError(f"A class has only {counts.min()} example(s); at least {MIN_PER_CLASS} are required to place it in "
                             "train, validation and test sets. Remove or merge rare classes.")
        strat = y
    try:
        rest, test = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=strat)
        rel = val_frac / (1 - test_frac)
        train, val = train_test_split(rest, test_size=rel, random_state=seed, stratify=None if strat is None else y[rest])
    except ValueError as e:
        raise SplitError(f"Could not split the data: {e}") from e
    split = {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}
    assert not (set(split["train"]) & set(split["test"])) and not (set(split["val"]) & set(split["test"]))
    return split


def make_split_strategy(y, task_type: str, strategy: str = "stratified", seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15,
                        groups=None, order=None) -> dict:
    """Split by strategy: 'stratified'/'random' (as make_split), 'group' (no group crosses partitions), 'temporal' (train < val < test in time)."""
    y = np.asarray(y)
    n = len(y)
    if strategy in ("stratified", "random"):
        return make_split(y, task_type, seed, val_frac, test_frac)
    if strategy == "group":
        if groups is None:
            raise SplitError("Group split needs a group column.")
        groups = np.asarray(groups)
        if len(np.unique(groups)) < 6:
            raise SplitError("Group split needs at least 6 distinct groups so that train, validation and test each get some.")
        idx = np.arange(n)
        rest, test = next(GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed).split(idx, y, groups))
        train_rel, val_rel = next(GroupShuffleSplit(n_splits=1, test_size=val_frac / (1 - test_frac), random_state=seed).split(rest, y[rest], groups[rest]))
        split = {"train": np.sort(rest[train_rel]), "val": np.sort(rest[val_rel]), "test": np.sort(test)}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if set(groups[split[a]]) & set(groups[split[b]]):
                raise SplitError("Internal error: a group appears in two partitions.")
        return split
    if strategy == "temporal":
        if order is None:
            raise SplitError("Temporal split needs a time column.")
        pos = np.argsort(np.asarray(order), kind="stable")
        n_test, n_val = int(round(n * test_frac)), int(round(n * val_frac))
        if n - n_test - n_val < 10:
            raise SplitError("Not enough rows for a temporal split.")
        return {"train": np.sort(pos[: n - n_test - n_val]), "val": np.sort(pos[n - n_test - n_val: n - n_test]), "test": np.sort(pos[n - n_test:])}
    raise SplitError(f"Unknown split strategy {strategy!r}")


def split_hash(split: dict) -> str:
    h = hashlib.sha256()
    for k in ("train", "val", "test"):
        h.update(k.encode())
        h.update(np.asarray(split[k], dtype=np.int64).tobytes())
    return h.hexdigest()
