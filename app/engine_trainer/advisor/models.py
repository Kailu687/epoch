"""The Advice data structure returned by the Epoch Advisor."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from typing import Any

TASK_TYPES = ("classification", "regression", "unknown")
SPLIT_STRATEGIES = ("stratified", "random", "group", "temporal")
BUDGET_MODES = ("express", "balanced", "deep")
OPT_PROFILES = ("balanced", "accuracy_first", "speed_first", "lightweight", "interpretable")
CONFIDENCES = ("high", "medium", "low")


@dataclass
class Advice:
    """Everything Epoch decided (or could not decide) for a dataset.

    All public fields are plain JSON types. ``_state`` is a private, non-serialised
    reference to the inputs so ``apply_answers`` can re-resolve; it is never written
    by ``to_dict``.
    """

    task_type: str = "unknown"
    target: str | None = None
    target_candidates: list = field(default_factory=list)   # [{column, score, reasons[]}]
    data_kind: str = "tabular"
    metric: str | None = None          # None only while the task is still unknown
    split_strategy: str = "random"
    group_column: str | None = None
    time_column: str | None = None
    budget_mode: str = "balanced"
    optimisation_profile: str = "balanced"
    exclude_columns: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    explanations: list = field(default_factory=list)
    questions: list = field(default_factory=list)   # [{id, text, options[]}]
    confidence: str = "low"
    _state: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        """JSON-serialisable dict of the public fields (deep-copied)."""
        return {f.name: copy.deepcopy(getattr(self, f.name)) for f in fields(self) if not f.name.startswith("_")}

    @classmethod
    def from_dict(cls, d: dict) -> "Advice":
        """Rebuild an Advice from ``to_dict`` output (without the private re-resolve state)."""
        known = {f.name for f in fields(cls) if not f.name.startswith("_")}
        return cls(**{k: copy.deepcopy(v) for k, v in d.items() if k in known})
