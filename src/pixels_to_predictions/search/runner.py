"""Subprocess orchestration: one training run per trial.

The runner launches ``python -m pixels_to_predictions.train --config-json ...`` as
a subprocess, captures stdout/stderr to a per-trial log file, and reads back the
``metrics.json`` the training script writes on exit.

Keeping each trial in its own subprocess is important: it insulates the overnight
loop from CUDA OOM / segfaults in any single trial.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
from contextlib import suppress
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING

from .experiment import Trial, TrialStatus

if TYPE_CHECKING:
    from pathlib import Path

    from ..config import RunConfig

logger = logging.getLogger("p2p.search.runner")


def run_trial(
    trial: Trial,
    run_cfg: RunConfig,
    log_dir: Path,
    timeout_s: float | None = None,
) -> Trial:
    """Run one trial to completion (or failure).

    Args:
        trial: the Trial record to mutate with status/metrics.
        run_cfg: the RunConfig to pass to ``train.main()``.
        log_dir: where this trial's stdout/stderr log will be written.
        timeout_s: optional per-trial wall-clock timeout. ``None`` means no timeout.

    Returns the mutated Trial.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{trial.trial_id}.log"
    trial.log_path = str(log_path)
    trial.started_at = datetime.now(tz=UTC).isoformat()
    trial.status = TrialStatus.RUNNING

    config_json = json.dumps(run_cfg.to_dict())
    cmd = [sys.executable, "-m", "pixels_to_predictions.train", "--config-json", config_json]

    t0 = perf_counter()
    env = os.environ.copy()
    # Force child stdout to be line-buffered so tail -f on the log is useful.
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w") as log_f:
        log_f.write(f"# trial {trial.trial_id} run_name={trial.run_name}\n")
        log_f.write(f"# cmd: {cmd}\n\n")
        log_f.flush()
        proc = subprocess.Popen(  # noqa: S603 — cmd is built from trusted internal sources
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            exit_code = proc.wait(timeout=timeout_s)
            trial.exit_code = exit_code
            trial.status = TrialStatus.COMPLETED if exit_code == 0 else TrialStatus.FAILED
        except subprocess.TimeoutExpired:
            logger.warning("Trial %s timed out after %.1fs; sending SIGTERM.", trial.trial_id, timeout_s)
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                exit_code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover -- escalate to SIGKILL
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                exit_code = proc.wait()
            trial.status = TrialStatus.KILLED
            trial.exit_code = exit_code

    trial.wall_clock_s = perf_counter() - t0
    trial.finished_at = datetime.now(tz=UTC).isoformat()

    # Read back metrics if the training script emitted them.
    metrics_path = run_cfg.run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            trial.metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            logger.exception("Could not parse %s", metrics_path)
    return trial
