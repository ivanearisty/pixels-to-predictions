"""Dataset loading, prompt formatting, and MCQ answer encoding.

The competition CSVs carry one row per question. Every row has:

    id, image_path, question, choices (JSON list), num_choices, answer,
    hint, lecture, task, grade, subject, topic, category, skill

Images live at ``data/images/{split}/{id}.png`` (note the ``images/`` dir is doubled
inside the provided zip — ``setup_data.py`` flattens that).

For SmolVLM, we render each sample as a single-turn chat:

    system: "Answer the multiple-choice question with only the letter ..."
    user:   [image] + question text + enumerated choices (+ optional hint, lecture)
    assistant: "A" | "B" | ... (gold label during training, generated at eval)

The MCQ label is always a single token; this keeps generation cheap and parsing
robust.
"""

from __future__ import annotations

import json
import string
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from PIL.Image import Image

    from .config import DataConfig


SYSTEM_PROMPT = (
    "You are a careful science tutor answering a multiple-choice question. "
    "Look at the image and read the question and all the choices. "
    "Reply with exactly one letter indicating the correct choice."
)

CHOICE_LETTERS = string.ascii_uppercase  # "ABCDEFGHIJ..." — supports up to 26 choices


@dataclass(frozen=True)
class MCQSample:
    """One multiple-choice question after preprocessing."""

    id: str
    image_path: Path
    question: str
    choices: list[str]
    hint: str | None
    lecture: str | None
    # Chain-of-thought walkthrough (train/val only -- absent on test).
    # Available as a future CoT-distillation signal; not used by the current loop.
    solution: str | None
    # Ground-truth answer as a 0-indexed integer. None for the test split.
    answer_index: int | None
    # Pedagogical metadata (subject/topic/category) for per-slice analysis.
    subject: str | None
    topic: str | None
    skill: str | None
    grade: str | None

    @property
    def num_choices(self) -> int:
        return len(self.choices)

    @property
    def answer_letter(self) -> str | None:
        """Gold answer as a letter (e.g. 'A'), or None if unlabeled."""
        if self.answer_index is None:
            return None
        return CHOICE_LETTERS[self.answer_index]


def _resolve_image_path(data_root: Path, image_subdir: str, split: str, sample_id: str) -> Path:
    """Return the absolute PNG path for a sample id.

    The provided zip stores images under ``images/images/<split>/<id>.png``; we flatten
    that on extraction so the canonical on-disk layout is ``data/images/<split>/<id>.png``.
    """
    return data_root / image_subdir / split / f"{sample_id}.png"


def load_split(cfg: DataConfig, split: str) -> list[MCQSample]:
    """Load and parse one CSV split into a list of MCQSample.

    Args:
        cfg: data configuration.
        split: ``"train"`` | ``"val"`` | ``"test"``.

    Returns:
        List of parsed samples. ``test`` samples will have ``answer_index=None``.
    """
    csv_path = cfg.root / {"train": cfg.train_csv, "val": cfg.val_csv, "test": cfg.test_csv}[split]
    df = pd.read_csv(csv_path)

    cap = cfg.max_train_samples if split == "train" else cfg.max_val_samples
    if cap is not None:
        df = df.head(cap)

    samples: list[MCQSample] = []
    for row in df.itertuples(index=False):
        raw_choices = row.choices
        # The CSV stores choices as a JSON-encoded list. If the cell is already a list
        # (possible with certain pandas dtypes), just use it.
        choices: list[str]
        if isinstance(raw_choices, str):
            choices = [str(c) for c in json.loads(raw_choices)]
        else:
            choices = [str(c) for c in list(raw_choices)]  # type: ignore[arg-type]

        sample_id = str(row.id)
        answer = getattr(row, "answer", None)
        answer_index = None if pd.isna(answer) else int(answer)

        samples.append(
            MCQSample(
                id=sample_id,
                image_path=_resolve_image_path(cfg.root, cfg.image_subdir, split, sample_id),
                question=str(row.question),
                choices=[str(c) for c in choices],
                hint=_row_opt(row, "hint"),
                lecture=_row_opt(row, "lecture"),
                solution=_row_opt(row, "solution"),
                answer_index=answer_index,
                subject=_row_opt(row, "subject"),
                topic=_row_opt(row, "topic"),
                skill=_row_opt(row, "skill"),
                grade=_row_opt(row, "grade"),
            ),
        )
    return samples


def _row_opt(row: object, field: str) -> str | None:
    """Return ``str(row.field)``, or None if the field is missing / NaN."""
    v = getattr(row, field, None)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v)


def format_user_turn(sample: MCQSample, *, include_hint: bool, include_lecture: bool) -> str:
    """Render the user-turn text for a MCQ sample.

    The image is attached separately (as a multimodal chat content item). Only the
    text component is returned here.
    """
    parts: list[str] = []
    if include_lecture and sample.lecture:
        parts.extend(["Lecture context:", sample.lecture, ""])
    parts.append(f"Question: {sample.question}")
    if include_hint and sample.hint:
        parts.extend(["", f"Hint: {sample.hint}"])
    parts.append("")
    parts.append("Choices:")
    for i, choice in enumerate(sample.choices):
        parts.append(f"  {CHOICE_LETTERS[i]}. {choice}")
    parts.append("")
    parts.append("Reply with a single letter, nothing else.")
    return "\n".join(parts)


def format_user_turn_with_caption(
    sample: MCQSample,
    caption: str | None,
    *,
    include_hint: bool,
    include_lecture: bool,
) -> str:
    r"""Render the user-turn text, optionally prepending a SmolVLM-generated caption.

    When ``caption`` is a non-empty string, the rendered text is prefixed with
    ``"[Image content: {caption}]\n\n"``. When ``caption`` is None or empty,
    this falls back to the standard :func:`format_user_turn` output.
    """
    base = format_user_turn(
        sample,
        include_hint=include_hint,
        include_lecture=include_lecture,
    )
    if caption:
        return f"[Image content: {caption}]\n\n" + base
    return base


def load_captions(path: Path) -> dict[str, str]:
    """Return a sample-id -> caption mapping, or an empty dict if the file is absent.

    The captions JSON is produced by ``scripts/generate_captions.py`` (SmolVLM
    self-captioning). Callers should treat a missing file as "no captions" and
    proceed with the regular prompt formatting.
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {str(k): str(v) for k, v in raw.items()}


def format_assistant_turn(sample: MCQSample, *, cot: bool = False) -> str:
    r"""Render the assistant-turn text for a MCQ sample.

    Two modes:
      - ``cot=False`` (default): the bare answer letter, e.g. ``"B"``. This is the
        original training target — a single token the collator learns to emit.
      - ``cot=True``: chain-of-thought distillation target,
        ``"{solution}\n\nThe answer is {letter}."``. The model is trained to first
        emit the step-by-step reasoning from the gold ``solution`` column, then
        declare the final answer in a parseable form. If ``solution`` is missing
        (e.g. test split or sparse rows), we fall back to the bare letter so the
        collator still has a valid target.

    The caller is responsible for ensuring ``sample.answer_letter`` is non-None
    (i.e. only invoke when building a labeled training/eval example).
    """
    if sample.answer_letter is None:
        msg = "format_assistant_turn requires a labeled sample (answer_letter is None)."
        raise ValueError(msg)
    if cot and sample.solution:
        return f"{sample.solution}\n\nThe answer is {sample.answer_letter}."
    return sample.answer_letter


def load_image(path: Path, image_size: int) -> Image:
    """Load a PNG, convert to RGB, and resize to ``image_size`` on the longest side.

    SmolVLM accepts non-square inputs, but normalizing to a fixed longest side keeps
    batch collation simple.
    """
    from PIL import Image as PILImage

    img = PILImage.open(path).convert("RGB")
    # Resize keeping aspect ratio, clamped to image_size on the longest side
    w, h = img.size
    scale = image_size / max(w, h)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, PILImage.Resampling.LANCZOS)
    return img
