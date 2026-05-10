"""Trial dataclass + JSONL serialization for the overnight optimizer.

One ``Trial`` represents one training subprocess: the config it ran under, the
metrics it produced, wall-clock time, and an overall status.
"""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field
from typing import Any


class TrialStatus(str, enum.Enum):
    """Lifecycle of a trial."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"  # early-stopped by ASHA or user SIGINT


@dataclass
class Trial:
    """One trial within a search run."""

    trial_id: str
    run_name: str
    config: dict[str, Any]  # serialised RunConfig
    status: TrialStatus = TrialStatus.PENDING
    metrics: dict[str, Any] = field(default_factory=dict)
    wall_clock_s: float | None = None
    started_at: str | None = None  # ISO timestamp
    finished_at: str | None = None  # ISO timestamp
    log_path: str | None = None
    exit_code: int | None = None

    def to_jsonl(self) -> str:
        """Serialise to a single JSONL line."""
        d = asdict(self)
        d["status"] = self.status.value
        return json.dumps(d, default=str)

    @classmethod
    def from_jsonl(cls, line: str) -> Trial:
        """Parse a single JSONL line back into a Trial."""
        d = json.loads(line)
        d["status"] = TrialStatus(d["status"])
        return cls(**d)

    @property
    def primary_metric(self) -> float:
        """Headline metric used for ranking (higher is better).

        Prefers validation accuracy; falls back to -train_loss when accuracy
        is unavailable (e.g. truncated ASHA rung).
        """
        if "accuracy" in self.metrics:
            return float(self.metrics["accuracy"])
        if "train_loss" in self.metrics:
            return -float(self.metrics["train_loss"])
        return float("-inf")
