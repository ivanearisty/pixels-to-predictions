"""Auto-discover trained-model logit .npz files and pick the best ensemble combination.

Reads all `<name>-val.npz` and matching `<name>-test.npz` from a logits directory.
For each non-empty subset of available models (up to a configurable cap), computes
the val accuracy of the summed-logit ensemble. Picks the subset that maximizes
val accuracy and writes the test submission for that subset.

Usage:
    python scripts/build_best_ensemble.py \
        --logits-dir results/logits \
        --out results/submissions/$(date +%Y%m%d-%H%M%S)-best-ensemble.csv \
        --max-subset-size 4
"""

from __future__ import annotations

import argparse
import itertools
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return z["ids"], z["logits"]


def align(base_ids: np.ndarray, ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
    order = {sid: i for i, sid in enumerate(ids)}
    idx = np.array([order[sid] for sid in base_ids])
    return logits[idx]


def ensemble_val_acc(
    models_logits: dict[str, np.ndarray],
    val_ids: np.ndarray,
    val_csv: pd.DataFrame,
) -> float:
    summed = np.zeros_like(next(iter(models_logits.values())))
    for lg in models_logits.values():
        summed = summed + lg
    nc_lookup = dict(zip(val_csv["id"].to_numpy(), val_csv["num_choices"].to_numpy(), strict=True))
    gold = dict(zip(val_csv["id"].to_numpy(), val_csv["answer"].to_numpy(), strict=True))
    correct = 0
    total = 0
    for sid, row in zip(val_ids, summed, strict=True):
        nc = int(nc_lookup[sid])
        masked = row.copy()
        masked[nc:] = -np.inf
        pred = int(np.argmax(masked))
        gold_ans = gold.get(sid)
        if gold_ans is not None:
            total += 1
            if int(gold_ans) == pred:
                correct += 1
    return correct / max(1, total)


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0912 -- orchestrator
    parser = argparse.ArgumentParser("build_best_ensemble")
    parser.add_argument("--logits-dir", type=Path, default=Path("results/logits"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-subset-size", type=int, default=4)
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=(),
        help="Variant names to skip (matches a model where its val.npz is named <variant>-val.npz).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional markdown ranking report path.",
    )
    args = parser.parse_args(argv)

    logits_dir = args.logits_dir
    val_files = sorted(p for p in logits_dir.glob("*-val.npz") if p.stem.removesuffix("-val") not in set(args.exclude))
    if not val_files:
        msg = f"no val.npz files found under {logits_dir}"
        raise FileNotFoundError(msg)

    variants = [p.stem.removesuffix("-val") for p in val_files]
    print(f"discovered {len(variants)} variants: {variants}")

    # Load all val logits, align by base_ids from the first file.
    base_ids_arr: np.ndarray | None = None
    val_logits_by_variant: dict[str, np.ndarray] = {}
    for v in variants:
        ids, lg = load_npz(logits_dir / f"{v}-val.npz")
        if base_ids_arr is None:
            base_ids_arr = ids
            val_logits_by_variant[v] = lg
        else:
            val_logits_by_variant[v] = align(base_ids_arr, ids, lg)

    if base_ids_arr is None:  # pragma: no cover -- impossible given the file check above
        msg = "no base ids could be loaded"
        raise RuntimeError(msg)
    val_csv = pd.read_csv(args.data_root / "val.csv")

    # Try every non-empty subset up to max_subset_size.
    results: list[tuple[float, tuple[str, ...]]] = []
    max_size = min(args.max_subset_size, len(variants))
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(variants, size):
            sub = {v: val_logits_by_variant[v] for v in combo}
            acc = ensemble_val_acc(sub, base_ids_arr, val_csv)
            results.append((acc, combo))

    results.sort(key=lambda t: -t[0])
    best_acc, best_combo = results[0]
    print(f"\nBEST: {best_combo} val_acc={best_acc:.4f}")
    print("Top 10 combinations:")
    for acc, combo in results[:10]:
        print(f"  {acc:.4f}  {' + '.join(combo)}")

    # Write test ensemble for the best combo.
    test_csv = pd.read_csv(args.data_root / "test.csv", usecols=["id", "num_choices"])
    nc_lookup = dict(zip(test_csv["id"].to_numpy(), test_csv["num_choices"].to_numpy(), strict=True))

    # Load test logits for best combo, align.
    test_base_ids: np.ndarray | None = None
    test_summed: np.ndarray | None = None
    for v in best_combo:
        test_path = logits_dir / f"{v}-test.npz"
        if not test_path.exists():
            print(f"WARNING: {test_path} missing -- skipping {v} from test ensemble")
            continue
        ids, lg = load_npz(test_path)
        if test_base_ids is None:
            test_base_ids = ids
            test_summed = lg.copy()
        else:
            aligned = align(test_base_ids, ids, lg)
            test_summed = test_summed + aligned

    if test_summed is None or test_base_ids is None:
        msg = "no test .npz files found for the best val combo -- can't write submission"
        raise FileNotFoundError(msg)

    rows: list[tuple[str, int]] = []
    for sid, row in zip(test_base_ids, test_summed, strict=True):
        nc = int(nc_lookup[sid])
        masked = row.copy()
        masked[nc:] = -np.inf
        rows.append((str(sid), int(np.argmax(masked))))

    sample_ss = pd.read_csv(args.data_root / "sample_submission.csv")
    pred_map = dict(rows)
    sample_ss["answer"] = sample_ss["id"].map(pred_map).fillna(0).astype(int)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample_ss[["id", "answer"]].to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(sample_ss)} rows)")
    print(f"submission combo: {best_combo}, val_acc={best_acc:.4f}")

    if args.report is not None:
        stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
        lines = [
            f"# Best-ensemble report -- {stamp}",
            "",
            f"- Variants discovered: {len(variants)}",
            f"- Subsets tried: up to size {max_size}",
            f"- **Winner**: `{' + '.join(best_combo)}` -> val_acc = **{best_acc:.4f}**",
            f"- Submission: `{args.out}`",
            "",
            "## Top 20 combinations",
            "",
            "| rank | val_acc | combo |",
            "|---:|---:|:---|",
        ]
        for i, (acc, combo) in enumerate(results[:20], start=1):
            lines.append(f"| {i} | {acc:.4f} | {' + '.join(combo)} |")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(lines))
        print(f"wrote ranking report to {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
