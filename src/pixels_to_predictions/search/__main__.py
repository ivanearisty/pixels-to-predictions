"""CLI for the overnight optimizer.

# Run a random search with an 8-hour wall-clock budget.
python -m pixels_to_predictions.search run \
        --space configs/search/overnight_v1.py \
        --strategy random \
        --budget 8h \
        --out logs/search/$(date +%Y%m%d-%H%M)

# Rebuild a report from an existing run.
python -m pixels_to_predictions.search report \
        --run logs/search/20260422-2300
"""

from __future__ import annotations

import argparse
import logging
import random
import signal
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from .runner import run_trial
from .scheduler import GridScheduler, RandomScheduler, Scheduler
from .space import SearchSpace, load_search_space
from .summary import render_summary
from .trials import append_trial, load_trials

if TYPE_CHECKING:
    from .experiment import Trial

logger = logging.getLogger("p2p.search")


def _parse_duration(s: str) -> float:
    """Parse strings like ``"30m"``, ``"8h"``, ``"90s"`` to seconds."""
    s = s.strip().lower()
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)


def _build_scheduler(strategy: str, space: SearchSpace, max_trials: int | None) -> Scheduler:
    if strategy == "random":
        return RandomScheduler(space=space, max_trials=max_trials)
    if strategy == "grid":
        return GridScheduler(space=space)
    msg = f"Unknown search strategy: {strategy!r}"
    raise ValueError(msg)


def _cmd_run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.jsonl"
    log_dir = out_dir / "logs"
    budget_s = _parse_duration(args.budget)

    space = load_search_space(Path(args.space))
    scheduler = _build_scheduler(args.strategy, space, args.max_trials)
    rng = random.Random(args.seed)

    logger.info("Search run dir: %s", out_dir)
    logger.info("Space: %s", args.space)
    logger.info("Strategy: %s  Budget: %.0fs  Seed: %s", args.strategy, budget_s, args.seed)

    t0 = perf_counter()
    completed: list[Trial] = []

    # Graceful SIGINT: finish the current trial's report, then exit.
    stop_flag = {"stop": False}

    def _on_sigint(_signum: int, _frame: object) -> None:
        logger.warning("SIGINT received — stopping after current trial.")
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        trial_idx = 0
        while True:
            if stop_flag["stop"]:
                break
            if perf_counter() - t0 > budget_s:
                logger.info("Budget exhausted (%.0fs). Stopping.", budget_s)
                break
            overrides = scheduler.next_overrides(rng)
            if overrides is None:
                logger.info("Scheduler exhausted (no more trials). Stopping.")
                break

            trial_idx += 1
            trial_id = f"t{trial_idx:04d}-{datetime.now(tz=UTC).strftime('%H%M%S')}"
            run_name = f"{out_dir.name}/{trial_id}"
            run_cfg = replace(
                space.apply_overrides(overrides),
                training=replace(space.apply_overrides(overrides).training, run_name=run_name),
            )

            from .experiment import Trial as _Trial

            trial = _Trial(trial_id=trial_id, run_name=run_name, config=run_cfg.to_dict())
            logger.info(
                "Starting trial %s (elapsed %.0fs/%.0fs) overrides=%s",
                trial_id,
                perf_counter() - t0,
                budget_s,
                overrides,
            )

            # Cap the trial at the remaining budget so a runaway single trial
            # can't eat the whole night.
            remaining = budget_s - (perf_counter() - t0)
            per_trial_timeout = args.trial_timeout
            if per_trial_timeout is not None:
                per_trial_timeout = min(per_trial_timeout, remaining)
            elif remaining > 0:
                per_trial_timeout = remaining

            trial = run_trial(trial, run_cfg, log_dir, timeout_s=per_trial_timeout)
            append_trial(trials_path, trial)
            scheduler.observe(trial)
            completed.append(trial)
            logger.info(
                "Finished trial %s status=%s primary=%.4f",
                trial_id,
                trial.status.value,
                trial.primary_metric,
            )
    finally:
        all_trials = load_trials(trials_path)
        report_path = render_summary(all_trials, out_dir, budget_s=budget_s)
        logger.info("Wrote summary to %s", report_path)

    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_dir = Path(args.run)
    trials = load_trials(run_dir / "trials.jsonl")
    out_path = render_summary(trials, run_dir)
    logger.info("Wrote %s (%d trials)", out_path, len(trials))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Top-level CLI dispatcher. Returns shell exit code."""
    parser = argparse.ArgumentParser("pixels_to_predictions.search")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a search.")
    p_run.add_argument("--space", type=str, required=True)
    p_run.add_argument("--out", type=str, required=True)
    p_run.add_argument("--strategy", choices=["random", "grid"], default="random")
    p_run.add_argument("--budget", type=str, default="8h")
    p_run.add_argument("--max-trials", type=int, default=None)
    p_run.add_argument("--trial-timeout", type=float, default=None)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.set_defaults(func=_cmd_run)

    p_report = sub.add_parser("report", help="Rebuild the markdown report from an existing run.")
    p_report.add_argument("--run", type=str, required=True)
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
