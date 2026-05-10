"""Overnight search-space v1.

A conservative first sweep: vary LoRA rank, alpha, learning rate, epochs, and
warmup while holding batch size and target modules constant. The product of
choice dimensions is intentionally small so grid search is also feasible.
"""

from __future__ import annotations

from pixels_to_predictions.config import RunConfig
from pixels_to_predictions.search.space import Choice, LogUniform, SearchSpace

SPACE = SearchSpace(
    base=RunConfig(),
    dimensions={
        "lora.r": Choice([4, 8, 16]),
        "lora.alpha": Choice([16, 32, 64]),
        "training.learning_rate": LogUniform(5e-5, 5e-4),
        "training.num_train_epochs": Choice([1.0, 2.0]),
        "training.warmup_ratio": Choice([0.0, 0.03, 0.1]),
    },
)
