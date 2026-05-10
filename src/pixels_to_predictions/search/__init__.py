"""Overnight hyperparameter optimizer, Karpathy-autoresearch style.

A simple time-budgeted orchestrator that:

1. Samples configs from a declarative ``SearchSpace``.
2. Spawns one training subprocess per trial, captured to its own log file.
3. Writes one JSONL line per trial as it finishes.
4. On budget exhaustion (or SIGINT), emits a markdown ranking report.

See ``python -m pixels_to_predictions.search --help`` for the CLI.
"""

from .experiment import Trial, TrialStatus
from .scheduler import GridScheduler, RandomScheduler, Scheduler
from .space import Choice, LogUniform, SearchSpace, Uniform

__all__ = [
    "Choice",
    "GridScheduler",
    "LogUniform",
    "RandomScheduler",
    "Scheduler",
    "SearchSpace",
    "Trial",
    "TrialStatus",
    "Uniform",
]
