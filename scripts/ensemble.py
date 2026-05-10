"""Average per-sample letter logits across multiple adapters and write a submission CSV.

Each input .npz must come from ``predict.py --save-logits`` and contain:
  - ``ids``: 1-D string array of sample IDs (length N)
  - ``logits``: (N, 5) float array; positions beyond a sample's num_choices are -inf

The script aligns rows by id, sums the logits (equivalent to averaging since we then
argmax), masks beyond each sample's num_choices using the test CSV's ``num_choices``
column, and writes a submission CSV in id-order matching ``sample_submission.csv``.

Usage:
    python scripts/ensemble.py \
        --logits results/logits/v1-test.npz results/logits/v3-test.npz \
        --out results/submissions/$(date +%Y%m%d-%H%M%S)-ensemble.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("ensemble")
    parser.add_argument("--logits", type=Path, nargs="+", required=True, help="One or more .npz files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--csv-name",
        type=str,
        default="test.csv",
        help="CSV to look up num_choices from. test.csv for test, val.csv for sanity-checking on val.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="sample_submission.csv",
        help="CSV whose id ordering the output should match. sample_submission.csv for Kaggle test.",
    )
    args = parser.parse_args(argv)

    # Load each logits file and align to a master id list.
    loaded: list[tuple[np.ndarray, np.ndarray]] = []
    base_ids: np.ndarray | None = None
    for p in args.logits:
        z = np.load(p, allow_pickle=False)
        ids = z["ids"]
        logits = z["logits"]
        if base_ids is None:
            base_ids = ids
        if not np.array_equal(np.sort(ids), np.sort(base_ids)):
            msg = f"id sets differ between {args.logits[0]} and {p}"
            raise ValueError(msg)
        # Reorder this file's logits to match base_ids ordering.
        order = {sid: i for i, sid in enumerate(ids)}
        idx = np.array([order[sid] for sid in base_ids])
        loaded.append((base_ids, logits[idx]))
        print(f"loaded {p}: shape={logits.shape}")

    if base_ids is None:
        msg = "no logit files provided"
        raise ValueError(msg)

    # Sum logits across adapters.
    summed = np.zeros_like(loaded[0][1])
    for _, lg in loaded:
        summed = summed + lg
    print(f"ensemble: averaging across {len(loaded)} adapters")

    # Apply num_choices mask from the appropriate CSV (test or val).
    nc_df = pd.read_csv(args.data_root / args.csv_name, usecols=["id", "num_choices"])
    nc_lookup = dict(zip(nc_df["id"].to_numpy(), nc_df["num_choices"].to_numpy(), strict=True))

    rows: list[tuple[str, int]] = []
    for sid, row in zip(base_ids, summed, strict=True):
        nc = int(nc_lookup[sid])
        masked = row.copy()
        masked[nc:] = -np.inf
        rows.append((str(sid), int(np.argmax(masked))))

    # Write in the reference CSV's id order.
    ref = pd.read_csv(args.data_root / args.reference)
    pred_map = dict(rows)

    # If reference is val.csv (or train.csv), it has a gold "answer" column;
    # report val accuracy before we overwrite it with our predictions.
    if "answer" in ref.columns and not ref["answer"].isna().any():
        gold = dict(zip(ref["id"].to_numpy(), ref["answer"].to_numpy(), strict=True))
        correct = sum(1 for sid, p in pred_map.items() if int(gold.get(sid, -1)) == int(p))
        total = sum(1 for sid in pred_map if sid in gold)
        if total:
            print(f"VAL accuracy: {correct}/{total} = {correct / total:.4f}")

    out_df = ref[["id"]].copy()
    out_df["answer"] = ref["id"].map(pred_map).fillna(0).astype(int)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out_df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
