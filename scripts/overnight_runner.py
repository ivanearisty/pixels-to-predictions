"""Overnight experiment driver.

Reads a JSON queue at ``logs/overnight/queue.json`` (a list of step dicts), executes
the next ``status="pending"`` step, captures its output, marks it ``completed`` or
``failed``, and exits. Designed to be re-invoked by an external scheduler (cron /
``ScheduleWakeup`` every N minutes); state is durable so successive invocations
just pick up where the last one stopped.

Step schema (one dict per stage in the queue):
    {
        "name": "v4_cot",                       # human-readable identifier
        "kind": "train" | "predict_val" | "predict_test" | "captions" | "shell",
        "cmd": ["python", "-m", ...],            # subprocess argv
        "log": "logs/v4_cot.train.log",         # path to capture stdout/stderr
        "depends_on": ["...other_step_name"],   # optional, must be completed first
        "status": "pending" | "running" | "completed" | "failed",
        "started_at": ..., "finished_at": ..., "exit_code": ...,
    }

Designed so adding new experiments overnight is a one-edit job to ``queue.json``.
The driver is *single-step* per invocation: it runs at most one step then exits.
This keeps each invocation short (works with ScheduleWakeup's session-resume model)
and lets you observe progress between fires.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

QUEUE_PATH = Path("logs/overnight/queue.json")
LOCK_PATH = Path("logs/overnight/runner.lock")


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def load_queue() -> list[dict[str, Any]]:
    if not QUEUE_PATH.exists():
        return []
    return json.loads(QUEUE_PATH.read_text())


def save_queue(queue: list[dict[str, Any]]) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.write_text(json.dumps(queue, indent=2))


def next_step(queue: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = {s["name"] for s in queue if s["status"] == "completed"}
    for s in queue:
        if s["status"] != "pending":
            continue
        deps = s.get("depends_on") or []
        if all(d in completed for d in deps):
            return s
    return None


def acquire_lock() -> bool:
    """Refuse to start if an earlier invocation is still running."""
    if LOCK_PATH.exists():
        # Stale lock if its PID is dead.
        try:
            pid = int(LOCK_PATH.read_text().strip())
            os.kill(pid, 0)
        except (OSError, ValueError):
            LOCK_PATH.unlink(missing_ok=True)
        else:
            return False
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(str(os.getpid()))
    return True


def release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def run_step(step: dict[str, Any]) -> int:
    log_path = Path(step["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    step["status"] = "running"
    step["started_at"] = _now()
    save_queue(load_queue_with_update(step))
    cmd = step["cmd"]
    print(f"[{_now()}] running {step['name']!r}: {cmd}", flush=True)
    with log_path.open("a") as f:
        f.write(f"\n# === run @ {_now()} ===\n# cmd: {cmd}\n\n")
        f.flush()
        proc = subprocess.run(  # noqa: S603 -- internal cmd
            cmd, stdout=f, stderr=subprocess.STDOUT, check=False,
        )
    step["exit_code"] = proc.returncode
    step["finished_at"] = _now()
    step["status"] = "completed" if proc.returncode == 0 else "failed"
    save_queue(load_queue_with_update(step))
    print(f"[{_now()}] step {step['name']!r} -> {step['status']} (exit {proc.returncode})", flush=True)
    return proc.returncode


def load_queue_with_update(updated: dict[str, Any]) -> list[dict[str, Any]]:
    queue = load_queue()
    for i, s in enumerate(queue):
        if s["name"] == updated["name"]:
            queue[i] = updated
            return queue
    queue.append(updated)
    return queue


def cmd_status() -> int:
    queue = load_queue()
    if not queue:
        print("(empty queue)")
        return 0
    counts: dict[str, int] = {}
    for s in queue:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    print(f"queue: {len(queue)} steps, " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    for s in queue:
        marker = {"completed": "✓", "failed": "✗", "running": "▶", "pending": "·"}.get(s["status"], "?")
        meta = ""
        if s["status"] == "completed":
            meta = f" exit={s.get('exit_code')}"
        elif s["status"] == "failed":
            meta = f" exit={s.get('exit_code')}  log={s['log']}"
        elif s["status"] == "running":
            meta = f" started={s.get('started_at')}"
        print(f"  {marker} {s['name']:30s} {s['status']:>10s}{meta}")
    return 0


def cmd_run() -> int:
    if not acquire_lock():
        print("another runner is active; exiting.")
        return 0
    try:
        queue = load_queue()
        step = next_step(queue)
        if step is None:
            print(f"[{_now()}] queue empty (nothing pending with satisfied deps).")
            return 0
        return run_step(step)
    finally:
        release_lock()


def cmd_run_loop() -> int:
    """Run pending steps sequentially until the queue is exhausted or one fails fatally.

    Designed for nohup-background invocation: the runner stays alive across the
    whole overnight session, so external polls just need to read ``queue.json``
    (no need to re-fire ``run`` per step). A ``failed`` step does NOT kill the
    loop -- we proceed to whatever's next, since later steps may be independent.
    """
    if not acquire_lock():
        print("another runner is active; exiting.")
        return 0
    try:
        while True:
            queue = load_queue()
            step = next_step(queue)
            if step is None:
                print(f"[{_now()}] queue exhausted, exiting run-loop.")
                return 0
            run_step(step)
    finally:
        release_lock()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("overnight_runner")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="Run the next pending step (single shot).")
    sub.add_parser("run-loop", help="Run pending steps until queue is exhausted.")
    sub.add_parser("status", help="Show queue state.")
    args = parser.parse_args(argv)
    if args.cmd == "status":
        return cmd_status()
    if args.cmd == "run-loop":
        return cmd_run_loop()
    return cmd_run()


if __name__ == "__main__":
    sys.exit(main())
