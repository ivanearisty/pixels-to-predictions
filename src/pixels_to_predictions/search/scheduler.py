"""Schedulers decide which config to try next.

All schedulers expose ``next_trial(rng) -> overrides | None``. A None return signals
"no more trials" (e.g. grid exhausted). ASHA additionally consumes trial results to
early-stop unpromising runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import random

    from .experiment import Trial
    from .space import SearchSpace


class Scheduler(Protocol):
    """Abstract scheduler protocol."""

    def next_overrides(self, rng: random.Random) -> dict[str, Any] | None: ...

    def observe(self, trial: Trial) -> None:
        """Hook for schedulers that use trial results (ASHA, Bayesian, etc.)."""
        ...


@dataclass
class RandomScheduler:
    """Unbounded random sampling from the search space."""

    space: SearchSpace
    max_trials: int | None = None
    _count: int = 0

    def next_overrides(self, rng: random.Random) -> dict[str, Any] | None:
        if self.max_trials is not None and self._count >= self.max_trials:
            return None
        self._count += 1
        return self.space.sample(rng)

    def observe(self, trial: Trial) -> None:  # noqa: ARG002 — protocol satisfied
        return


@dataclass
class GridScheduler:
    """Enumerate the cartesian product of Choice dimensions once."""

    space: SearchSpace
    _iter: Any = None

    def __post_init__(self) -> None:
        self._iter = iter(self.space.grid())

    def next_overrides(self, rng: random.Random) -> dict[str, Any] | None:  # noqa: ARG002
        return next(self._iter, None)

    def observe(self, trial: Trial) -> None:  # noqa: ARG002
        return


@dataclass
class ASHAScheduler:
    """Successive halving with random search.

    Generates random trials; early-stops trials whose metric at rung ``k`` is in
    the bottom fraction ``eta ** -1`` compared to their rung peers. A lightweight
    single-process version — enough for overnight runs on one GPU.
    """

    space: SearchSpace
    max_trials: int = 20
    rung_steps: list[int] = field(default_factory=lambda: [100, 400, 1600])
    eta: float = 3.0
    _history_by_rung: dict[int, list[float]] = field(default_factory=dict)
    _count: int = 0

    def next_overrides(self, rng: random.Random) -> dict[str, Any] | None:
        if self._count >= self.max_trials:
            return None
        self._count += 1
        return self.space.sample(rng)

    def observe(self, trial: Trial) -> None:
        """Record trial's primary metric at the last completed rung."""
        steps_completed = int(trial.metrics.get("step", 0))
        for rung in self.rung_steps:
            if steps_completed >= rung:
                self._history_by_rung.setdefault(rung, []).append(trial.primary_metric)

    def should_stop(self, trial: Trial, step: int) -> bool:
        """Return True if ``trial`` should be killed at ``step`` per ASHA rules."""
        if step not in self.rung_steps:
            return False
        history = self._history_by_rung.get(step, [])
        if len(history) < 2:
            return False
        history_sorted = sorted(history, reverse=True)
        cutoff_idx = max(1, int(len(history_sorted) / self.eta))
        cutoff = history_sorted[cutoff_idx - 1]
        return trial.primary_metric < cutoff
