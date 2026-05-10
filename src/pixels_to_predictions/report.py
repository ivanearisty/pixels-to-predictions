"""Aggregate per-run metrics into a markdown table + plot.

Used both by the overnight optimizer (for an end-of-night summary) and by ad-hoc
ablation studies in ``scripts/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


@dataclass
class RunSummary:
    """One row in the aggregate report table."""

    run_name: str
    accuracy: float | None
    trainable_params: int | None
    train_loss: float | None
    wall_clock_s: float | None
    config: dict[str, object]


def load_run_summaries(run_dirs: Iterable[Path]) -> list[RunSummary]:
    """Read ``metrics.json`` + ``config.json`` from each run directory."""
    out: list[RunSummary] = []
    for rd in run_dirs:
        metrics_path = rd / "metrics.json"
        config_path = rd / "config.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        out.append(
            RunSummary(
                run_name=rd.name,
                accuracy=metrics.get("accuracy"),
                trainable_params=metrics.get("trainable_params"),
                train_loss=metrics.get("train_loss"),
                wall_clock_s=metrics.get("wall_clock_s"),
                config=config,
            ),
        )
    return out


def render_markdown_table(summaries: list[RunSummary]) -> str:
    """Format run summaries as a single-glance markdown ranking table."""
    # Sort by accuracy desc, then by wall-clock asc.
    summaries = sorted(
        summaries,
        key=lambda s: (-(s.accuracy or -1.0), s.wall_clock_s or float("inf")),
    )
    header = "| rank | run | accuracy | trainable | loss | wall (s) |\n"
    sep = "|---:|:---|---:|---:|---:|---:|\n"
    rows = [
        f"| {i} | {s.run_name} | "
        f"{(s.accuracy or 0):.4f} | "
        f"{(s.trainable_params or 0):,} | "
        f"{(s.train_loss or 0):.4f} | "
        f"{(s.wall_clock_s or 0):.1f} |"
        for i, s in enumerate(summaries, start=1)
    ]
    return header + sep + "\n".join(rows)
