"""End-of-run markdown report generation for the overnight optimizer."""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .experiment import Trial


def render_summary(trials: list[Trial], out_dir: Path, budget_s: float | None = None) -> Path:
    """Write a markdown report of all trials to ``out_dir / report.md``.

    Returns the path to the rendered report.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.md"

    lines: list[str] = []
    lines.append(f"# Overnight search report — `{out_dir.name}`")
    lines.append("")
    lines.append(f"- **Total trials:** {len(trials)}")
    status_counts = Counter(t.status.value for t in trials)
    lines.append(
        "- **Status breakdown:** " + ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())),
    )
    if budget_s is not None:
        lines.append(f"- **Budget used:** {sum(t.wall_clock_s or 0 for t in trials):.1f}s / {budget_s:.0f}s")
    lines.append("")

    # Ranking table (only completed trials, sorted by primary metric desc).
    completed = [t for t in trials if t.status.value == "completed" and t.metrics]
    if completed:
        completed.sort(key=lambda t: t.primary_metric, reverse=True)
        lines.append("## Ranking")
        lines.append("")
        lines.append("| # | trial | primary | accuracy | train_loss | wall (s) | key config |")
        lines.append("|---:|:---|---:|---:|---:|---:|:---|")
        for i, t in enumerate(completed, start=1):
            acc = t.metrics.get("accuracy", "—")
            loss = t.metrics.get("train_loss", "—")
            cfg = _shorten_config(t.config)
            lines.append(
                f"| {i} | {t.trial_id} | {t.primary_metric:.4f} | {acc} | {loss} | {t.wall_clock_s or 0:.1f} | {cfg} |",
            )
        lines.append("")

        # Best trial deep dive.
        best = completed[0]
        lines.append("## Best trial")
        lines.append("")
        lines.append(f"- **id:** `{best.trial_id}`")
        lines.append(f"- **run_name:** `{best.run_name}`")
        lines.append(f"- **primary metric:** `{best.primary_metric:.4f}`")
        lines.append(f"- **log:** `{best.log_path}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(best.config, indent=2))
        lines.append("```")
        lines.append("")

        # Per-dimension sensitivity (top-k vs bottom-k on each varied hyperparam).
        varied = _find_varied_dims(completed)
        if varied:
            lines.append("## Per-hyperparameter sensitivity (top-quartile vs bottom-quartile mean)")
            lines.append("")
            lines.append("| dim | top-Q mean | bot-Q mean | delta |")
            lines.append("|:---|---:|---:|---:|")
            n = max(1, len(completed) // 4)
            top_q = completed[:n]
            bot_q = completed[-n:]
            for dim in varied:
                top_vals = [_get_path(t.config, dim) for t in top_q]
                bot_vals = [_get_path(t.config, dim) for t in bot_q]
                top_mean = _mean_if_numeric(top_vals)
                bot_mean = _mean_if_numeric(bot_vals)
                delta = (top_mean - bot_mean) if (top_mean is not None and bot_mean is not None) else None
                lines.append(
                    f"| `{dim}` | {_fmt(top_mean)} | {_fmt(bot_mean)} | {_fmt(delta)} |",
                )
            lines.append("")

    # Failures appendix.
    failures = [t for t in trials if t.status.value in {"failed", "killed"}]
    if failures:
        lines.append("## Failures")
        lines.append("")
        lines.extend(f"- `{t.trial_id}` ({t.status.value}, exit={t.exit_code}) — `{t.log_path}`" for t in failures)
        lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path


def _shorten_config(config: dict[str, object]) -> str:
    """Compact one-line rendering of the key hyperparameters."""
    try:
        lora = config.get("lora", {})
        training = config.get("training", {})
        parts: list[str] = []
        if isinstance(lora, dict):
            parts.append(f"r={lora.get('r')}")
            parts.append(f"a={lora.get('alpha')}")
        if isinstance(training, dict):
            parts.append(f"lr={training.get('learning_rate'):.1e}")
            parts.append(f"ep={training.get('num_train_epochs')}")
        return " ".join(parts)
    except (KeyError, TypeError, ValueError):  # pragma: no cover
        return ""


def _find_varied_dims(trials: list[Trial]) -> list[str]:
    """Return the dotted-path config keys whose value varies across trials."""
    seen: dict[str, set[object]] = {}
    for t in trials:
        for path, value in _flatten(t.config).items():
            seen.setdefault(path, set()).add(repr(value))
    return [k for k, vs in seen.items() if len(vs) > 1]


def _flatten(d: object, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = d
    return out


def _get_path(d: dict[str, object], path: str) -> object:
    cur: object = d
    for seg in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(seg)
        else:
            return None
    return cur


def _mean_if_numeric(values: list[object]) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return sum(nums) / len(nums) if nums else None


def _fmt(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.4g}"
