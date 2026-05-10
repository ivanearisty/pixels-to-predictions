"""Declarative search spaces for the overnight optimizer.

A ``SearchSpace`` maps dotted-path hyperparameter names (e.g. ``"lora.r"``,
``"training.learning_rate"``) to distributions. Configs are produced by:

1. Sampling one value per dimension.
2. Deep-merging those values into a copy of the base ``RunConfig``.

Example search space (saved to ``configs/search/overnight_v1.py``)::

    from pixels_to_predictions.config import RunConfig
    from pixels_to_predictions.search.space import SearchSpace, Choice, LogUniform

    SPACE = SearchSpace(
        base=RunConfig(),
        dimensions={
            "lora.r":                       Choice([4, 8, 16, 32]),
            "training.learning_rate":       LogUniform(1e-5, 1e-3),
            "training.num_train_epochs":    Choice([1, 2, 3]),
            "training.warmup_ratio":        Choice([0.0, 0.03, 0.1]),
        },
    )

The CLI loads a search space by importing the file and looking for a module-level
``SPACE`` binding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import random
    from collections.abc import Iterable
    from pathlib import Path

    from ..config import RunConfig


class Distribution(Protocol):
    """A 1D hyperparameter distribution."""

    def sample(self, rng: random.Random) -> Any: ...

    def grid_values(self) -> list[Any]:
        """Return values for a grid sweep. ``Uniform``/``LogUniform`` raise."""
        ...


@dataclass(frozen=True)
class Choice:
    """Uniform random choice from a fixed list."""

    values: list[Any]

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.values)

    def grid_values(self) -> list[Any]:
        return list(self.values)


@dataclass(frozen=True)
class Uniform:
    """Uniform real in [low, high]."""

    low: float
    high: float

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)

    def grid_values(self) -> list[Any]:  # pragma: no cover
        msg = "Uniform cannot be enumerated for grid search; use Choice instead."
        raise NotImplementedError(msg)


@dataclass(frozen=True)
class LogUniform:
    """Log-uniform real in [low, high]. Good for learning rates, weight decay."""

    low: float
    high: float

    def sample(self, rng: random.Random) -> float:
        return math.exp(rng.uniform(math.log(self.low), math.log(self.high)))

    def grid_values(self) -> list[Any]:  # pragma: no cover
        msg = "LogUniform cannot be enumerated for grid search; use Choice instead."
        raise NotImplementedError(msg)


@dataclass(frozen=True)
class SearchSpace:
    """A base RunConfig plus a set of dimensions to vary."""

    base: RunConfig
    dimensions: dict[str, Distribution] = field(default_factory=dict)

    def sample(self, rng: random.Random) -> dict[str, Any]:
        """Return a {dotted_path: value} dict of sampled overrides."""
        return {name: dist.sample(rng) for name, dist in self.dimensions.items()}

    def grid(self) -> Iterable[dict[str, Any]]:
        """Yield all cartesian combinations for a grid sweep."""
        import itertools

        names = list(self.dimensions.keys())
        value_lists = [self.dimensions[n].grid_values() for n in names]
        for combo in itertools.product(*value_lists):
            yield dict(zip(names, combo, strict=True))

    def apply_overrides(self, overrides: dict[str, Any]) -> RunConfig:
        """Return a new RunConfig with ``overrides`` merged into ``self.base``.

        Dotted paths may be 2 segments deep (``"lora.r"``) matching the nested
        dataclass structure of ``RunConfig``. Single-segment paths target the top
        level (unused in practice).
        """
        # Group overrides by top-level section ("lora", "training", etc.)
        by_section: dict[str, dict[str, Any]] = {}
        for path, value in overrides.items():
            head, _, tail = path.partition(".")
            if not tail:
                msg = f"Top-level overrides not supported: {path!r}"
                raise ValueError(msg)
            by_section.setdefault(head, {})[tail] = value

        # Build the new RunConfig by replacing each touched sub-dataclass.
        new_sections: dict[str, Any] = {}
        for section_name, patch in by_section.items():
            current = getattr(self.base, section_name)
            new_sections[section_name] = replace(current, **patch)

        return replace(self.base, **new_sections)


def load_search_space(path: Path) -> SearchSpace:
    """Load a Python file that exports a module-level ``SPACE`` binding."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("search_space_module", str(path))
    if spec is None or spec.loader is None:
        msg = f"Could not import search space from {path}"
        raise ImportError(msg)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["search_space_module"] = mod
    spec.loader.exec_module(mod)
    space = getattr(mod, "SPACE", None)
    if not isinstance(space, SearchSpace):
        msg = f"{path} must define a module-level SPACE: SearchSpace"
        raise TypeError(msg)
    return space
