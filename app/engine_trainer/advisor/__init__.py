"""Epoch Advisor ("Let Epoch decide"): deterministic, offline micro-AI. See README.md."""
from .engine import advise, apply_answers, describe_advice
from .models import Advice

__all__ = ["Advice", "advise", "apply_answers", "describe_advice"]
