"""Validation-set accuracy + per-subject breakdown.

The competition leaderboard is raw accuracy, but per-subject / per-topic splits
are critical for understanding where the model wins or loses. This module writes
both an overall number and a breakdown table.

Usage:
    python -m pixels_to_predictions.evaluate \
        --checkpoint outputs/baseline/checkpoint-final \
        --out results/ablations/baseline-val.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger("p2p.evaluate")


@dataclass
class SliceAccuracy:
    """Accuracy on a subset of the val set (per-subject, per-topic, etc.)."""

    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


@dataclass
class EvalReport:
    """Aggregate evaluation report ready to dump as JSON."""

    checkpoint: str
    overall: SliceAccuracy = field(default_factory=SliceAccuracy)
    by_subject: dict[str, SliceAccuracy] = field(default_factory=dict)
    by_topic: dict[str, SliceAccuracy] = field(default_factory=dict)
    by_num_choices: dict[int, SliceAccuracy] = field(default_factory=dict)
    num_samples: int = 0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict suitable for ablations/*.json."""
        return {
            "checkpoint": self.checkpoint,
            "num_samples": self.num_samples,
            "accuracy": self.overall.accuracy,
            "correct": self.overall.correct,
            "total": self.overall.total,
            "by_subject": {k: asdict(v) | {"accuracy": v.accuracy} for k, v in self.by_subject.items()},
            "by_topic": {k: asdict(v) | {"accuracy": v.accuracy} for k, v in self.by_topic.items()},
            "by_num_choices": {str(k): asdict(v) | {"accuracy": v.accuracy} for k, v in self.by_num_choices.items()},
        }


def score_predictions(
    predictions: dict[str, int],
    gold: list[tuple[str, int, str | None, str | None, int]],
    checkpoint: str = "",
) -> EvalReport:
    """Compute an EvalReport from model predictions and gold labels.

    Args:
        predictions: map of sample_id -> predicted 0-indexed answer.
        gold: list of (sample_id, gold_answer_index, subject, topic, num_choices).
        checkpoint: path to the checkpoint, logged in the report.
    """
    report = EvalReport(checkpoint=checkpoint)
    by_subj: dict[str, SliceAccuracy] = defaultdict(SliceAccuracy)
    by_topic: dict[str, SliceAccuracy] = defaultdict(SliceAccuracy)
    by_nc: dict[int, SliceAccuracy] = defaultdict(SliceAccuracy)

    for sample_id, gold_ans, subject, topic, num_choices in gold:
        pred = predictions.get(sample_id)
        if pred is None:
            continue
        correct = int(pred == gold_ans)
        report.overall.total += 1
        report.overall.correct += correct
        if subject is not None:
            by_subj[subject].total += 1
            by_subj[subject].correct += correct
        if topic is not None:
            by_topic[topic].total += 1
            by_topic[topic].correct += correct
        by_nc[num_choices].total += 1
        by_nc[num_choices].correct += correct

    report.by_subject = dict(by_subj)
    report.by_topic = dict(by_topic)
    report.by_num_choices = dict(by_nc)
    report.num_samples = report.overall.total
    return report


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("pixels_to_predictions.evaluate")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Produce a val-set EvalReport for a given checkpoint. Returns exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = _parse_cli(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # TODO(scaffold): wire predict-on-val + score_predictions here once predict.py is implemented.
    stub_report = EvalReport(checkpoint=str(args.checkpoint))
    args.out.write_text(json.dumps(stub_report.to_dict(), indent=2))
    logger.info("Wrote stub EvalReport to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
