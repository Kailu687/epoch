"""Improve an existing model with more data: compatibility/drift checks, combined retraining against the SAME locked test set,
and an honest old-versus-new comparison."""
from .compat import check_compatibility  # noqa: F401
