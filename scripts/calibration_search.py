"""Inference-only calibration search over the per-variant val logits.

Tries a grid of (per-model weights x low-confidence-fallback threshold x disagree
fallback). Ranks all combinations by val accuracy, prints the top configurations,
and writes a test submission for the val-best combo.

No GPU/training required -- operates on pre-saved .npz logits in
``results/logits/``.

Strategies (all composable):
  * weighted ensemble: sum_i w_i * logits_i, then masked argmax
  * low-conf fallback: when softmax-max-prob over valid letters < threshold,
    override with v1's per-sample argmax
  * disagree fallback: when v2/v3/v4_cot all give DIFFERENT argmaxes,
    override with v1's per-sample argmax

Usage:
    python scripts/calibration_search.py
"""

from __future__ import annotations

import itertools
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

LOGITS_DIR = Path("results/logits")
DATA_ROOT = Path("data")
OUT_DIR = Path("results")


def discover_variants() -> list[str]:
    """Find all variants with both -val.npz and -test.npz on disk."""
    val_files = {p.stem.removesuffix("-val") for p in LOGITS_DIR.glob("*-val.npz")}
    test_files = {p.stem.removesuffix("-test") for p in LOGITS_DIR.glob("*-test.npz")}
    return sorted(val_files & test_files)


VARIANTS = discover_variants()


def load_aligned(name: str, base_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(LOGITS_DIR / f"{name}-val.npz", allow_pickle=False)
    ids, logits = z["ids"], z["logits"]
    if base_ids is None:
        return ids, logits
    order = {sid: i for i, sid in enumerate(ids)}
    idx = np.array([order[sid] for sid in base_ids])
    return base_ids, logits[idx]


def predictions_from_logits(logits: np.ndarray, num_choices: np.ndarray) -> np.ndarray:
    """Argmax per row, masked to per-sample num_choices. Returns int array of length N."""
    n = len(num_choices)
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        k = int(num_choices[i])
        row = logits[i, :].copy()
        row[k:] = -np.inf
        out[i] = int(row.argmax())
    return out


def softmax_max_prob(logits: np.ndarray, num_choices: np.ndarray) -> np.ndarray:
    """For each row, softmax over the first k entries; return the max probability."""
    n = len(num_choices)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        k = int(num_choices[i])
        row = logits[i, :k]
        m = row.max()
        e = np.exp(row - m)
        p = e / e.sum()
        out[i] = float(p.max())
    return out


def build_summed(weights: dict[str, float], variant_logits: dict[str, np.ndarray]) -> np.ndarray:
    summed = None
    for v, w in weights.items():
        if w == 0:
            continue
        contrib = w * variant_logits[v]
        summed = contrib if summed is None else summed + contrib
    if summed is None:
        msg = "all-zero weights"
        raise ValueError(msg)
    return summed


def evaluate(
    *,
    weights: dict[str, float],
    variant_logits: dict[str, np.ndarray],
    variant_argmax: dict[str, np.ndarray],
    num_choices: np.ndarray,
    gold: np.ndarray,
    fallback_threshold: float | None = None,
    disagree_fallback_trio: tuple[str, ...] | None = None,
    fallback_to: str = "v1",
) -> tuple[int, np.ndarray]:
    """Compute (correct_count, predictions) for one strategy combination."""
    summed = build_summed(weights, variant_logits)
    preds = predictions_from_logits(summed, num_choices)
    if fallback_threshold is not None:
        max_p = softmax_max_prob(summed, num_choices)
        low = max_p < fallback_threshold
        preds = np.where(low, variant_argmax[fallback_to], preds)
    if disagree_fallback_trio is not None:
        trio_preds = np.stack([variant_argmax[v] for v in disagree_fallback_trio], axis=1)
        n_unique = np.array([len(set(row.tolist())) for row in trio_preds])
        all_disagree = n_unique == len(disagree_fallback_trio)
        preds = np.where(all_disagree, variant_argmax[fallback_to], preds)
    correct = int((preds == gold).sum())
    return correct, preds


def main() -> int:  # noqa: C901 -- single-shot driver
    val = pd.read_csv(DATA_ROOT / "val.csv")
    test = pd.read_csv(DATA_ROOT / "test.csv")

    # Establish base id ordering from v1's val.npz; align everyone else.
    base_ids = np.load(LOGITS_DIR / "v1-val.npz", allow_pickle=False)["ids"]
    variant_logits: dict[str, np.ndarray] = {}
    for v in VARIANTS:
        _, lg = load_aligned(v, base_ids)
        variant_logits[v] = lg

    # Align gold answers and num_choices to base_ids.
    val_lookup_nc = dict(zip(val["id"].to_numpy(), val["num_choices"].to_numpy(), strict=True))
    val_lookup_ans = dict(zip(val["id"].to_numpy(), val["answer"].to_numpy(), strict=True))
    num_choices = np.array([val_lookup_nc[sid] for sid in base_ids])
    gold = np.array([val_lookup_ans[sid] for sid in base_ids], dtype=np.int64)
    n = len(base_ids)

    # Cache each variant's per-sample argmax (with num_choices masking).
    variant_argmax = {v: predictions_from_logits(variant_logits[v], num_choices) for v in VARIANTS}
    print("=== individual val accuracy (sanity) ===")
    for v in VARIANTS:
        c = int((variant_argmax[v] == gold).sum())
        print(f"  {v:<20s} {c:>4d}/{n} = {c / n:.4f}")
    print()

    # Search grid:
    #   weights for top-4 variants in [0, 0.5, 1, 1.5, 2]; v5/v6 always 0 (they hurt)
    #   fallback_threshold in [None, 0.40, 0.50, 0.60]
    #   disagree_fallback_trio in [None, ("v2","v3","v4_cot")]
    weight_choices = [0.0, 0.5, 1.0, 1.5, 2.0]
    fallback_thresholds: list[float | None] = [None, 0.40, 0.50, 0.60]
    disagree_options: list[tuple[str, ...] | None] = [None, ("v2", "v3", "v4_cot")]
    top_variants = ["v1", "v2", "v3", "v4_cot"]

    results: list[dict] = []
    for combo in itertools.product(weight_choices, repeat=len(top_variants)):
        if all(w == 0 for w in combo):
            continue
        weights = dict(zip(top_variants, combo, strict=True))
        for fb in fallback_thresholds:
            for trio in disagree_options:
                try:
                    correct, _ = evaluate(
                        weights=weights,
                        variant_logits=variant_logits,
                        variant_argmax=variant_argmax,
                        num_choices=num_choices,
                        gold=gold,
                        fallback_threshold=fb,
                        disagree_fallback_trio=trio,
                    )
                except ValueError:
                    continue
                results.append({
                    "weights": weights,
                    "fallback_threshold": fb,
                    "disagree_trio": trio,
                    "correct": correct,
                    "acc": correct / n,
                })

    results.sort(key=lambda r: -r["acc"])
    print(f"=== ran {len(results):,} combinations; top 25 ===")
    print(f"{'rank':>4} {'val_acc':>8} {'weights':>40} {'fb':>5} {'disagree':>20}")
    for i, r in enumerate(results[:25], start=1):
        wstr = ",".join(f"{v}={w}" for v, w in r["weights"].items() if w > 0)
        fb = r["fallback_threshold"] or "—"
        trio = "+".join(r["disagree_trio"]) if r["disagree_trio"] else "—"
        print(f"{i:>4} {r['acc']:>.4f}  {wstr:>40}  {fb!s:>5}  {trio:>20}")

    # Identify the val-best and write a test submission for it.
    best = results[0]
    print()
    print(f"=== BEST: val_acc={best['acc']:.4f} ===")
    print(f"  weights: {best['weights']}")
    print(f"  fallback_threshold: {best['fallback_threshold']}")
    print(f"  disagree_trio: {best['disagree_trio']}")

    # Repeat the best strategy on TEST.
    test_logits: dict[str, np.ndarray] = {}
    test_base_ids = np.load(LOGITS_DIR / "v1-test.npz", allow_pickle=False)["ids"]
    for v in VARIANTS:
        z = np.load(LOGITS_DIR / f"{v}-test.npz", allow_pickle=False)
        order = {sid: i for i, sid in enumerate(z["ids"])}
        idx = np.array([order[sid] for sid in test_base_ids])
        test_logits[v] = z["logits"][idx]

    test_nc = dict(zip(test["id"].to_numpy(), test["num_choices"].to_numpy(), strict=True))
    test_num_choices = np.array([test_nc[sid] for sid in test_base_ids])
    test_argmax = {v: predictions_from_logits(test_logits[v], test_num_choices) for v in VARIANTS}

    summed = build_summed(best["weights"], test_logits)
    preds = predictions_from_logits(summed, test_num_choices)
    if best["fallback_threshold"] is not None:
        max_p = softmax_max_prob(summed, test_num_choices)
        low = max_p < best["fallback_threshold"]
        preds = np.where(low, test_argmax["v1"], preds)
    if best["disagree_trio"] is not None:
        trio_preds = np.stack([test_argmax[v] for v in best["disagree_trio"]], axis=1)
        n_unique = np.array([len(set(row.tolist())) for row in trio_preds])
        all_disagree = n_unique == len(best["disagree_trio"])
        preds = np.where(all_disagree, test_argmax["v1"], preds)

    pred_map = dict(zip(test_base_ids, preds.tolist(), strict=True))
    sample_ss = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    sample_ss["answer"] = sample_ss["id"].map(pred_map).fillna(0).astype(int)

    stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    out_csv = OUT_DIR / "submissions" / f"{stamp}-calibrated-best.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sample_ss[["id", "answer"]].to_csv(out_csv, index=False)
    print(f"\nWrote test submission: {out_csv}")
    print(f"Test answer distribution: {dict(sample_ss['answer'].value_counts().sort_index())}")

    # Markdown report.
    report = OUT_DIR / "ablations" / f"{stamp}-calibration-search.md"
    lines = [
        f"# Calibration search -- {stamp}",
        "",
        f"- Combinations evaluated: {len(results):,}",
        f"- Best val_acc: **{best['acc']:.4f}**",
        f"- Best weights: `{best['weights']}`",
        f"- Best fallback_threshold: `{best['fallback_threshold']}`",
        f"- Best disagree_trio: `{best['disagree_trio']}`",
        f"- Submission: `{out_csv}`",
        "",
        "## Top 25",
        "",
        "| rank | val_acc | weights | fb | disagree-trio |",
        "|---:|---:|:---|:---:|:---|",
    ]
    for i, r in enumerate(results[:25], start=1):
        wstr = ", ".join(f"{v}={w}" for v, w in r["weights"].items() if w > 0)
        fb = r["fallback_threshold"] or "—"
        trio = "+".join(r["disagree_trio"]) if r["disagree_trio"] else "—"
        lines.append(f"| {i} | {r['acc']:.4f} | `{wstr}` | {fb} | {trio} |")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    print(f"Wrote report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
