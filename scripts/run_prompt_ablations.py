"""Run a fixed set of prompt-format ablations against val for one trained checkpoint.

Each ablation is a different combination of ``--include-{hint,lecture,metadata}`` and
``--prompt-style`` flags passed to ``pixels_to_predictions.predict``. The val accuracy
for every ablation is parsed out of its log file and dumped as a markdown table at the
end so we can pick the strongest prompt for the final test submission.

Usage:
    python scripts/run_prompt_ablations.py --checkpoint outputs/v2/checkpoint-final
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

# Each ablation is (name, list_of_extra_flags). Names appear in the report table.
ABLATIONS: list[tuple[str, list[str]]] = [
    ("baseline_trained",         []),
    ("drop_lecture",             ["--no-lecture"]),
    ("drop_metadata",            ["--no-metadata"]),
    ("minimal_no_lec_no_meta",   ["--no-lecture", "--no-metadata"]),
    ("answer_is",                ["--prompt-style", "answer_is"]),
    ("answer_is_minimal",        ["--no-lecture", "--no-metadata", "--prompt-style", "answer_is"]),
]


def _parse_val_acc(log: str) -> float | None:
    m = re.search(r"VAL accuracy:\s+(\d+)/(\d+)\s*=\s*([0-9.]+)", log)
    if not m:
        return None
    return float(m.group(3))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("run_prompt_ablations")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/ablations"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/ablations"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args(argv)

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    rows: list[dict[str, object]] = []

    for name, extra in ABLATIONS:
        out_csv = args.logs_dir / f"{stamp}-{name}.csv"
        log_path = args.logs_dir / f"{stamp}-{name}.log"
        cmd = [
            sys.executable,
            "-m", "pixels_to_predictions.predict",
            "--checkpoint", str(args.checkpoint),
            "--out", str(out_csv),
            "--split", "val",
            "--data-root", str(args.data_root),
            "--batch-size", str(args.batch_size),
            "--image-size", str(args.image_size),
            *extra,
        ]
        print(f"=== {name} ===  flags: {extra}")
        t0 = perf_counter()
        with log_path.open("w") as f:
            f.write(f"# cmd: {cmd}\n\n")
            f.flush()
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)  # noqa: S603 -- internal cmd
        dt = perf_counter() - t0
        log = log_path.read_text()
        acc = _parse_val_acc(log)
        print(f"  acc={acc!r}  wall={dt:.1f}s  log={log_path}")
        rows.append({"name": name, "flags": " ".join(extra), "val_acc": acc, "wall_s": dt})

    # Markdown report
    rows.sort(key=lambda r: -(r["val_acc"] or -1.0))
    md_path = args.results_dir / f"{stamp}-prompt_ablations.md"
    json_path = args.results_dir / f"{stamp}-prompt_ablations.json"
    lines = [
        f"# Prompt ablation report — `{args.checkpoint}`",
        "",
        f"- Run timestamp (UTC): `{stamp}`",
        f"- Total ablations: {len(rows)}",
        "",
        "| rank | name | flags | val_acc | wall (s) |",
        "|---:|:---|:---|---:|---:|",
    ]
    lines.extend(
        f"| {i} | {r['name']} | `{r['flags']}` | "
        f"{(r['val_acc'] or 0):.4f} | {(r['wall_s']):.1f} |"
        for i, r in enumerate(rows, start=1)
    )
    md_path.write_text("\n".join(lines))
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {md_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
