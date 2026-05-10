"""Trial registry: JSONL-backed read/append for the overnight optimizer.

One line per trial. Reads are O(n) but trials are small (hundreds at most in a
single overnight run), so this keeps the code simple and the file human-readable
for later inspection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .experiment import Trial

if TYPE_CHECKING:
    from pathlib import Path


def append_trial(jsonl_path: Path, trial: Trial) -> None:
    """Append one trial to the JSONL registry. Creates the file if missing."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a") as f:
        f.write(trial.to_jsonl() + "\n")


def load_trials(jsonl_path: Path) -> list[Trial]:
    """Read all trials from a JSONL registry. Returns empty list if the file doesn't exist."""
    if not jsonl_path.exists():
        return []
    out: list[Trial] = []
    with jsonl_path.open() as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                out.append(Trial.from_jsonl(stripped))
    return out
