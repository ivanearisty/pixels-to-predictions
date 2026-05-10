r"""Slice-level error analysis for the v1 LoRA model on the validation set.

Reconstructs v1's predicted answers from saved letter logits (matching the per-sample
``num_choices`` masking ensemble.py uses) and breaks down accuracy across data slices
to find the worst-performing buckets we should target with v2/v3.

Slices computed:
    * ``num_choices`` (2/3/4/5)
    * ``subject`` (natural / social / language science)
    * ``topic`` (~13 topics)
    * ``grade`` (grade1 ... grade12)
    * has-rich-hint (hint non-null and >100 chars -- proxy for text-only solvable)
    * image archetype (heuristic on PIL ``size`` -- diagram / photo / wide-banner)

Outputs a markdown report with one table per slice, sorted worst-first, and a
"biggest opportunities" preamble (top 5 high-volume + low-accuracy buckets).

Usage:
    python scripts/failure_analysis.py
    python scripts/failure_analysis.py --logits results/logits/v1-val.npz \
        --csv data/val.csv --out results/ablations/failure_analysis_v1.md
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable


HINT_RICH_THRESHOLD_CHARS = 100
DIAGRAM_SQUARE_SIZES: frozenset[tuple[int, int]] = frozenset({(162, 162), (202, 202), (200, 200)})
WIDE_ASPECT_THRESHOLD = 2.0
PHOTO_ASPECT_LOW = 0.7
PHOTO_ASPECT_HIGH = 1.4
DIAGRAM_RES_LIMIT = 260  # px on the long side for a "tiny diagram"
OPPORTUNITY_MIN_N = 30  # ignore tiny slices when ranking opportunities
TOP_OPPORTUNITIES = 5


@dataclass(frozen=True)
class SliceRow:
    """One row of a per-slice accuracy breakdown.

    Attributes:
        slice_type: Name of the slice family (e.g. ``"topic"``).
        slice_value: Bucket label inside that family (e.g. ``"physics"``).
        n: Number of validation samples in the bucket.
        correct: Number of samples v1 predicted correctly.
        accuracy: ``correct / n``, in ``[0, 1]``.
    """

    slice_type: str
    slice_value: str
    n: int
    correct: int
    accuracy: float


def reconstruct_predictions(npz_path: Path, num_choices: dict[str, int]) -> dict[str, int]:
    """Reconstruct v1's argmax predictions from saved letter logits.

    Mirrors ensemble.py: positions ``>= num_choices`` are masked to ``-inf`` before
    argmax, so a sample with ``num_choices=3`` can only land on indices 0, 1, or 2.

    Args:
        npz_path: Path to ``v1-val.npz`` with ``ids`` and ``logits`` arrays.
        num_choices: Map from sample id to its ``num_choices``.

    Returns:
        Mapping from sample id (str) to predicted 0-indexed answer (int).
    """
    z = np.load(npz_path, allow_pickle=False)
    ids = z["ids"]
    logits = z["logits"].astype(np.float64, copy=True)

    preds: dict[str, int] = {}
    for sid, row in zip(ids, logits, strict=True):
        nc = num_choices[str(sid)]
        masked = row.copy()
        masked[nc:] = -np.inf
        preds[str(sid)] = int(np.argmax(masked))
    return preds


def _image_archetype(image_path: Path) -> str:
    """Bucket an image into one of {diagram, photo, wide_banner, other} via PIL size.

    Args:
        image_path: Absolute path to the PNG/JPEG image on disk.

    Returns:
        Coarse archetype label suitable as a dataframe slice value.
    """
    try:
        with Image.open(image_path) as img:
            w, h = img.size
    except (FileNotFoundError, OSError):
        return "unreadable"

    if (w, h) in DIAGRAM_SQUARE_SIZES:
        return "diagram_fixed_square"
    if w == h and max(w, h) <= DIAGRAM_RES_LIMIT:
        return "diagram_small_square"

    aspect = w / h if h else 0.0
    if aspect >= WIDE_ASPECT_THRESHOLD:
        return "wide_banner"
    if PHOTO_ASPECT_LOW <= aspect <= PHOTO_ASPECT_HIGH and max(w, h) >= DIAGRAM_RES_LIMIT:
        return "photo_natural"
    return "other"


def _has_rich_hint(hint: object) -> bool:
    """Return whether the ``hint`` field is a long text passage (not NaN, > threshold chars).

    Args:
        hint: Raw value from the ``hint`` column (may be ``NaN``, str, or pd.NA).

    Returns:
        ``True`` when ``hint`` is a string with more than ``HINT_RICH_THRESHOLD_CHARS``
        characters; ``False`` otherwise.
    """
    if hint is None or (isinstance(hint, float) and np.isnan(hint)):
        return False
    if pd.isna(hint):
        return False
    return isinstance(hint, str) and len(hint) > HINT_RICH_THRESHOLD_CHARS


def annotate(df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """Add ``has_rich_hint`` and ``image_archetype`` slice columns to the val frame.

    Args:
        df: Loaded ``val.csv`` frame; must contain ``hint`` and ``image_path``.
        data_root: Path that ``image_path`` is relative to (typically ``data/``).

    Returns:
        The input frame with two new columns added (mutated copy).
    """
    out = df.copy()
    out["has_rich_hint"] = out["hint"].apply(_has_rich_hint).map(
        {True: "rich_hint", False: "no_rich_hint"},
    )
    out["image_archetype"] = out["image_path"].apply(
        lambda p: _image_archetype(data_root / str(p)),
    )
    return out


def _accuracy_for(df: pd.DataFrame, slice_type: str, key: str | Callable[[pd.Series], str]) -> list[SliceRow]:
    """Compute per-bucket accuracy rows for one slice family.

    Args:
        df: Annotated val frame containing at least ``answer`` and ``pred`` columns
            plus the column referenced by ``key``.
        slice_type: Name to attach to every emitted ``SliceRow``.
        key: Either a column name to group by, or a function applied row-wise that
            returns the bucket label.

    Returns:
        One ``SliceRow`` per non-empty bucket, sorted ascending by accuracy.
    """
    bucket = df.apply(key, axis=1) if callable(key) else df[key]
    rows: list[SliceRow] = []
    for value, group in df.groupby(bucket):
        n = len(group)
        correct = int((group["pred"] == group["answer"]).sum())
        rows.append(SliceRow(slice_type, str(value), n, correct, correct / n if n else 0.0))
    rows.sort(key=lambda r: (r.accuracy, -r.n))
    return rows


def render_table(rows: list[SliceRow], header: str) -> list[str]:
    """Render one slice's rows as a sorted markdown table.

    Args:
        rows: Pre-sorted ``SliceRow`` list (caller decides ordering).
        header: Title for the markdown section ("by topic", "by grade", ...).

    Returns:
        List of markdown lines, including the section heading and table.
    """
    out = [
        f"## {header}",
        "",
        "| slice | n | correct | accuracy |",
        "|:---|---:|---:|---:|",
    ]
    out.extend(f"| {r.slice_value} | {r.n} | {r.correct} | {r.accuracy:.4f} |" for r in rows)
    out.append("")
    return out


def biggest_opportunities(all_rows: list[SliceRow], min_n: int, top_k: int) -> list[SliceRow]:
    """Pick the top-K largest underperforming slices across every slice family.

    "Largest underperforming" means: among slices with at least ``min_n`` samples,
    rank by ``(accuracy ascending, n descending)`` so a low-accuracy bucket with
    many samples beats a slightly-lower-accuracy bucket with very few samples.

    Args:
        all_rows: Concatenation of every slice family's rows.
        min_n: Minimum bucket size to be considered.
        top_k: How many opportunities to return.

    Returns:
        Up to ``top_k`` rows, ordered worst-first.
    """
    candidates = [r for r in all_rows if r.n >= min_n]
    candidates.sort(key=lambda r: (r.accuracy, -r.n))
    return candidates[:top_k]


def main(argv: list[str] | None = None) -> int:
    """Run the failure analysis and write the markdown report.

    Args:
        argv: Optional argv override for testing; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser("failure_analysis")
    parser.add_argument("--logits", type=Path, default=Path("results/logits/v1-val.npz"))
    parser.add_argument("--csv", type=Path, default=Path("data/val.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/ablations/failure_analysis_v1.md"))
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)
    nc_lookup = dict(zip(df["id"].astype(str), df["num_choices"].astype(int), strict=True))
    preds = reconstruct_predictions(args.logits, nc_lookup)
    df["pred"] = df["id"].astype(str).map(preds).astype(int)
    df["answer"] = df["answer"].astype(int)
    df = annotate(df, args.data_root)

    overall_n = len(df)
    overall_correct = int((df["pred"] == df["answer"]).sum())
    overall_acc = overall_correct / overall_n if overall_n else 0.0

    slice_specs: list[tuple[str, str, str | Callable[[pd.Series], str]]] = [
        ("by num_choices", "num_choices", "num_choices"),
        ("by subject", "subject", "subject"),
        ("by topic", "topic", "topic"),
        ("by grade", "grade", "grade"),
        ("by hint richness", "has_rich_hint", "has_rich_hint"),
        ("by image archetype", "image_archetype", "image_archetype"),
    ]

    sections: list[tuple[str, list[SliceRow]]] = [
        (header, _accuracy_for(df, slice_type, key)) for header, slice_type, key in slice_specs
    ]
    all_rows: list[SliceRow] = [row for _, rows in sections for row in rows]
    top_ops = biggest_opportunities(all_rows, OPPORTUNITY_MIN_N, TOP_OPPORTUNITIES)

    lines: list[str] = [
        f"# Failure analysis -- v1 on val (`{args.logits.name}`)",
        "",
        f"- Source logits: `{args.logits}`",
        f"- Source labels: `{args.csv}`",
        f"- Overall accuracy: **{overall_correct}/{overall_n} = {overall_acc:.4f}**",
        f"- Slice families analyzed: {len(sections)}",
        "",
        "## Biggest opportunities (top 5 high-volume, low-accuracy slices)",
        "",
        f"Slices restricted to those with n >= {OPPORTUNITY_MIN_N}; ranked by accuracy "
        "ascending then size descending.",
        "",
        "| rank | slice_type | slice | n | correct | accuracy |",
        "|---:|:---|:---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {i} | {r.slice_type} | {r.slice_value} | {r.n} | {r.correct} | {r.accuracy:.4f} |"
        for i, r in enumerate(top_ops, start=1)
    )
    lines.append("")

    for header, rows in sections:
        lines.extend(render_table(rows, header))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"overall: {overall_correct}/{overall_n} = {overall_acc:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
