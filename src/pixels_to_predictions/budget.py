"""Enforce the 5 M trainable-parameter competition cap.

The competition rules make the parameter budget a hard disqualifier. We want a
single choke-point that every training entry goes through, so that removing or
bypassing it is a one-line git diff that review can flag.

Typical usage::

    model = build_model(run_cfg)
    audit = audit_parameters(model)
    audit.assert_under_budget()
    print(audit.summary())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import TRAINABLE_PARAM_BUDGET

if TYPE_CHECKING:
    from torch import nn


class BudgetExceededError(RuntimeError):
    """Raised when the trainable-parameter count exceeds the competition cap."""


@dataclass(frozen=True)
class ParameterAudit:
    """Parameter-count breakdown for a model.

    Attributes:
        total: Total parameters (frozen + trainable).
        trainable: Parameters with ``requires_grad=True``.
        by_component: Per-top-level-module trainable-param counts; useful for
            spotting when LoRA accidentally attached to the vision encoder.
        budget: The cap this audit is checked against.
    """

    total: int
    trainable: int
    by_component: dict[str, int]
    budget: int = TRAINABLE_PARAM_BUDGET

    @property
    def trainable_fraction(self) -> float:
        return self.trainable / self.total if self.total > 0 else 0.0

    @property
    def under_budget(self) -> bool:
        return self.trainable <= self.budget

    @property
    def headroom(self) -> int:
        return self.budget - self.trainable

    def assert_under_budget(self) -> None:
        """Raise ``BudgetExceededError`` if trainable params exceed the cap."""
        if not self.under_budget:
            msg = (
                f"Trainable parameter budget exceeded: "
                f"{self.trainable:,} > {self.budget:,} (over by "
                f"{self.trainable - self.budget:,}). "
                f"Reduce LoRA rank, drop target modules, or narrow scope."
            )
            raise BudgetExceededError(msg)

    def summary(self) -> str:
        """Human-readable single-block summary."""
        lines = [
            f"Total params:      {self.total:>13,}",
            f"Trainable params:  {self.trainable:>13,}   ({100 * self.trainable_fraction:5.2f}% of total)",
            f"Budget:            {self.budget:>13,}   (headroom: {self.headroom:,})",
            "By component (trainable-param count):",
        ]
        lines.extend(
            f"  {name:<30s} {count:>13,}"
            for name, count in sorted(self.by_component.items(), key=lambda kv: -kv[1])
            if count > 0
        )
        return "\n".join(lines)


def audit_parameters(model: nn.Module) -> ParameterAudit:
    """Walk the model and count total + trainable params, grouped by top-level module."""
    total = 0
    trainable = 0
    by_component: dict[str, int] = {}

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
            # Group by first path segment: "model.text_model.layers.0.q_proj" -> "model"
            # but we want "text_model" or "vision_model"; we take the second segment
            # when the first is "model" / "base_model".
            segments = name.split(".")
            comp = segments[1] if segments[0] in {"model", "base_model"} and len(segments) > 1 else segments[0]
            by_component[comp] = by_component.get(comp, 0) + n

    return ParameterAudit(total=total, trainable=trainable, by_component=by_component)
