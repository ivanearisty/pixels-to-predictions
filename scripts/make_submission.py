"""Produce a Kaggle-ready submission.csv from a trained checkpoint.

Thin wrapper around ``pixels_to_predictions.predict`` that stamps the output
file name with the checkpoint basename + timestamp so overwrites don't silently
clobber a working submission.

Usage:
    python scripts/make_submission.py --checkpoint outputs/baseline/checkpoint-final
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from pixels_to_predictions.predict import main as predict_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("make_submission")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/submissions"))
    parser.add_argument("--label", type=str, default=None, help="Extra label to append to filename.")
    args, extra = parser.parse_known_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    tag = args.checkpoint.name
    label = f"-{args.label}" if args.label else ""
    out_csv = args.out_dir / f"{stamp}-{tag}{label}.csv"

    return predict_main(
        ["--checkpoint", str(args.checkpoint), "--out", str(out_csv), *extra],
    )


if __name__ == "__main__":
    sys.exit(main())
