"""Run a trained SmolVLM+LoRA checkpoint over the test split → submission.csv.

Loads the base SmolVLM model, attaches the trained LoRA adapter from the given
checkpoint dir, iterates the test set, generates 4 tokens greedily, and writes
the parsed answer index to a CSV in the format Kaggle expects.

Usage:
    python -m pixels_to_predictions.predict \
        --checkpoint outputs/baseline/checkpoint-final \
        --out results/submissions/baseline.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from .config import DataConfig
from .data import (
    CHOICE_LETTERS,
    MCQSample,
    format_user_turn_with_caption,
    load_captions,
    load_image,
    load_split,
)

logger = logging.getLogger("p2p.predict")


PROMPT_TAILS = {
    # Default tail used during training. The model has been LoRA-tuned on this
    # exact phrase, so this is the in-distribution choice.
    "trained": "Reply with a single letter, nothing else.",
    # Completion-mode framing: encourages the model to treat the next token as
    # the answer. Out-of-distribution at inference but commonly improves MCQ
    # for instruction-tuned models.
    "answer_is": "The correct answer is:",
    # No explicit instruction; relies entirely on the chat template + choices
    # block to anchor the answer. Riskiest, included for completeness.
    "none": "",
}


def _build_messages(
    sample: MCQSample,
    data_cfg: DataConfig,
    *,
    prompt_style: str = "trained",
    caption: str | None = None,
) -> list[dict[str, Any]]:
    r"""User-turn messages (no assistant turn) for greedy generation.

    ``prompt_style`` swaps the closing instruction line:
      - ``"trained"``   — exactly what the model saw during fine-tuning
      - ``"answer_is"`` — completion-mode rewording
      - ``"none"``      — no closing instruction

    When ``caption`` is non-empty, ``"[Image content: {caption}]\n\n"`` is
    prepended to the user turn (before any subject metadata).
    """
    user_text = format_user_turn_with_caption(
        sample,
        caption,
        include_hint=data_cfg.include_hint,
        include_lecture=data_cfg.include_lecture,
    )
    if data_cfg.include_metadata:
        meta_bits = [b for b in (sample.subject, sample.topic, sample.grade) if b]
        if meta_bits:
            user_text = f"[Subject: {' / '.join(meta_bits)}]\n\n" + user_text

    # Replace the trained closing line if a different style is requested.
    trained_tail = PROMPT_TAILS["trained"]
    if prompt_style != "trained":
        new_tail = PROMPT_TAILS[prompt_style]
        if user_text.endswith(trained_tail):
            user_text = user_text[: -len(trained_tail)].rstrip() + (
                ("\n\n" + new_tail) if new_tail else ""
            )

    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def parse_letter_to_index(letter: str, num_choices: int) -> int:
    """Map a generated string to a 0-indexed answer.

    Robustness order:
      1. First A/B/C/... char that's in range [0, num_choices).
      2. If no valid letter found, fall back to 0 (a missing row would
         disqualify the entire submission, so we always return *something*).
    """
    valid = set(CHOICE_LETTERS[:num_choices])
    for ch in letter.strip().upper():
        if ch in valid:
            return CHOICE_LETTERS.index(ch)
    return 0


_COT_ANSWER_RE = re.compile(r"answer\s+is\s*[:\.]?\s*([A-E])", re.IGNORECASE)


def parse_cot_letter(text: str, num_choices: int) -> int:
    r"""Parse a chain-of-thought generation into a 0-indexed answer.

    CoT-trained outputs look like ``"...reasoning...\n\nThe answer is B."``.
    Strategy:
      1. Search for the trained marker phrase ``"answer is X"`` (case-insensitive,
         optional ``:`` or ``.`` between ``is`` and the letter). This is the
         strongest signal — it's the exact form the model was trained on.
      2. If the marker is missing (model hallucinated, truncated, etc.), fall
         back to the *last* valid letter in [A, ..., A+num_choices-1] anywhere
         in the text. The last letter is preferred over the first because CoT
         output frequently mentions choice letters mid-reasoning before
         settling on a final one.
      3. If still nothing, return 0 — same defensive default as
         ``parse_letter_to_index`` to keep the submission row count intact.
    """
    valid = set(CHOICE_LETTERS[:num_choices])
    match = _COT_ANSWER_RE.search(text)
    if match:
        candidate = match.group(1).upper()
        if candidate in valid:
            return CHOICE_LETTERS.index(candidate)
    # Fallback: scan from the end for the most recent valid letter.
    for ch in reversed(text.upper()):
        if ch in valid:
            return CHOICE_LETTERS.index(ch)
    return 0


def load_model_and_processor(
    checkpoint: Path,
    base_model_id: str,
    *,
    disable_image_split: bool = False,
) -> tuple[Any, Any]:
    """Load base SmolVLM + LoRA adapter, return on cuda.

    Sets ``padding_side='left'`` on the tokenizer: decoder-only models generate
    from the rightmost non-pad token, so right-padding corrupts the generation
    context for shorter prompts in a batch. Must be set before generation, not
    before training (training uses right-padding + label masking).

    If ``disable_image_split=True``, also disables the processor's
    image-splitting step (~64 visual tokens per image instead of ~320). Must
    match the value used during training, otherwise inference is OOD.
    """
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint if (checkpoint / "tokenizer.json").exists() else base_model_id)
    processor.tokenizer.padding_side = "left"
    if disable_image_split:
        processor.image_processor.do_image_splitting = False
    base = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, str(checkpoint))
    model.to("cuda").eval()
    return model, processor


def _letter_token_ids(processor: Any, max_choices: int = 5) -> list[int]:
    """Return the token IDs the model actually emits for each answer letter.

    SmolLM2's BPE tokenizer (used by SmolVLM) merges leading spaces with the
    following character: " A" / " B" / ... are *single tokens distinct from*
    the bare "A" / "B" / ... tokens. Since our chat template ends with
    "Assistant: " and the model's training target is " B<end_of_utterance>",
    the answer token at the prediction position is the *space-letter* token,
    not the bare letter. Using bare-letter IDs at inference reads logits at
    the wrong vocabulary positions and silently regresses accuracy.

    We encode " A", " B", ..., " E" (each with a leading space) and take the
    last token of each encoding -- that's the actual answer-position token.
    """
    out: list[int] = []
    for letter in CHOICE_LETTERS[:max_choices]:
        ids = processor.tokenizer.encode(" " + letter, add_special_tokens=False)
        if not ids:
            msg = f"Empty tokenization for letter {letter!r}"
            raise RuntimeError(msg)
        out.append(ids[-1])
    return out


@torch.no_grad()
def predict_split(
    model: Any,
    processor: Any,
    samples: list[MCQSample],
    data_cfg: DataConfig,
    *,
    batch_size: int = 4,
    max_new_tokens: int = 4,
    prompt_style: str = "trained",
    use_cot: bool = False,
    captions: dict[str, str] | None = None,
) -> dict[str, int]:
    """Generate answer letters for every sample, return {sample_id -> 0-indexed answer}.

    When ``use_cot=True`` the model is expected to emit a full reasoning trace
    followed by ``"The answer is X."``. We override ``max_new_tokens`` to 512 so
    there is room for the reasoning, and parse via ``parse_cot_letter`` instead
    of the bare-letter parser. The ``max_length`` cap is also raised to 8192 to
    match the bumped training-time sequence budget.
    """
    if use_cot:
        max_new_tokens = max(max_new_tokens, 512)
    parse_fn = parse_cot_letter if use_cot else parse_letter_to_index
    encoder_max_length = 8192 if use_cot else 4096
    out: dict[str, int] = {}
    pbar = tqdm(range(0, len(samples), batch_size), desc="predict", unit="batch")
    for start in pbar:
        batch = samples[start : start + batch_size]
        texts = [
            processor.apply_chat_template(
                _build_messages(
                    s,
                    data_cfg,
                    prompt_style=prompt_style,
                    caption=(captions.get(s.id) if captions else None),
                ),
                add_generation_prompt=True,
            )
            for s in batch
        ]
        images = [[load_image(s.image_path, data_cfg.image_size)] for s in batch]
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=encoder_max_length,
        ).to("cuda")
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        # Strip the prompt portion: each generation is prompt_ids + new_ids
        prompt_len = inputs["input_ids"].size(1)
        new_ids = gen_ids[:, prompt_len:]
        decoded = processor.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        for s, txt in zip(batch, decoded, strict=True):
            out[s.id] = parse_fn(txt, s.num_choices)
    return out


@torch.no_grad()
def predict_split_logits(
    model: Any,
    processor: Any,
    samples: list[MCQSample],
    data_cfg: DataConfig,
    *,
    batch_size: int = 4,
    prompt_style: str = "trained",
    save_logits_path: Path | None = None,
    captions: dict[str, str] | None = None,
    logit_prefix: str = "",
) -> dict[str, int]:
    """Logit-based prediction: argmax over letter-token logits at the answer position.

    Compared to ``predict_split`` this:
      - Runs a single forward pass per sample (no generation loop).
      - Reads logits at position -1 (the position right after "Assistant: " plus
        any ``logit_prefix``).
      - Works because we set ``padding_side='left'`` so every sample's content
        ends at the same index.
      - Restricts argmax to ``[A, B, C, ..., D]`` valid choices for the sample.
      - Has no decoding/parsing step -- avoids text-generation idiosyncrasies.

    ``logit_prefix`` is a raw string appended after ``apply_chat_template`` to
    move the prediction position past the literal "Assistant: " into something
    the model expects to be followed by an answer letter. Example:
    ``logit_prefix=" The answer is"`` makes the model predict the next token
    after "...Assistant: The answer is" -- which is the answer letter for any
    model trained on the standard MCQ template (essential for CoT-distilled
    models, where position -1 without a prefix is the start of the solution).

    If ``save_logits_path`` is given, writes a .npz with two arrays:
      - ``ids``: array of sample-id strings
      - ``logits``: (N, max_choices) float array; positions beyond a sample's
        ``num_choices`` are filled with ``-inf``.
    """
    letter_ids = _letter_token_ids(processor, max_choices=5)
    letter_id_tensor = torch.tensor(letter_ids, device="cuda")
    out: dict[str, int] = {}
    saved_ids: list[str] = []
    saved_logits: list[list[float]] = []  # one row per sample, length 5

    pbar = tqdm(range(0, len(samples), batch_size), desc="predict[logits]", unit="batch")
    for start in pbar:
        batch = samples[start : start + batch_size]
        texts = [
            processor.apply_chat_template(
                _build_messages(
                    s,
                    data_cfg,
                    prompt_style=prompt_style,
                    caption=(captions.get(s.id) if captions else None),
                ),
                add_generation_prompt=True,
            )
            for s in batch
        ]
        if logit_prefix:
            texts = [t + logit_prefix for t in texts]
        images = [[load_image(s.image_path, data_cfg.image_size)] for s in batch]
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to("cuda")

        # Single forward pass; no generation.
        outputs = model(**inputs)
        last_logits = outputs.logits[:, -1, :]  # (batch, vocab)
        # Slice down to the 5 candidate letter positions.
        letter_logits = last_logits.index_select(dim=1, index=letter_id_tensor).float().cpu()

        for i, s in enumerate(batch):
            row = letter_logits[i].clone()
            # Mask out letters beyond this sample's num_choices.
            row[s.num_choices :] = float("-inf")
            out[s.id] = int(row.argmax().item())
            if save_logits_path is not None:
                saved_ids.append(s.id)
                saved_logits.append(row.tolist())

    if save_logits_path is not None:
        import numpy as np

        save_logits_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_logits_path,
            ids=np.array(saved_ids),
            logits=np.array(saved_logits, dtype=np.float32),
        )
        logger.info("Saved per-sample letter logits to %s", save_logits_path)
    return out


def main(argv: list[str] | None = None) -> int:
    """Generate a Kaggle submission CSV from a trained checkpoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser("pixels_to_predictions.predict")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--include-hint", action="store_true", default=True)
    parser.add_argument("--no-hint", dest="include_hint", action="store_false")
    parser.add_argument("--include-lecture", action="store_true", default=True)
    parser.add_argument("--no-lecture", dest="include_lecture", action="store_false")
    parser.add_argument("--include-metadata", action="store_true", default=True)
    parser.add_argument("--no-metadata", dest="include_metadata", action="store_false")
    parser.add_argument("--prompt-style", choices=list(PROMPT_TAILS.keys()), default="trained")
    parser.add_argument(
        "--inference-mode",
        choices=["generate", "logits"],
        default="logits",
        help="generate=greedy text + parse; logits=single-pass argmax over letter tokens",
    )
    parser.add_argument(
        "--save-logits",
        type=Path,
        default=None,
        help="If set (and --inference-mode=logits), dump per-sample 5-letter logits to this .npz file.",
    )
    parser.add_argument("--base-model", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument(
        "--use-cot",
        action="store_true",
        default=False,
        help=(
            "Chain-of-thought inference: generate up to 512 tokens and parse the answer "
            "from 'The answer is X'. Only meaningful with --inference-mode=generate; the "
            "logits path is unaffected."
        ),
    )
    parser.add_argument(
        "--use-captions",
        action="store_true",
        default=False,
        help=(
            "Prepend a SmolVLM-self-generated image caption to the user turn. "
            "Captions are read from --captions-path. Errors loudly if the file is missing."
        ),
    )
    parser.add_argument(
        "--captions-path",
        type=Path,
        default=Path("data/captions.json"),
        help="JSON file mapping sample-id -> caption (built by scripts/generate_captions.py).",
    )
    parser.add_argument(
        "--logit-prefix",
        type=str,
        default="",
        help=(
            "Append this raw string after apply_chat_template (logits mode only). "
            "Use ' The answer is' for CoT-distilled models so position -1 is the "
            "answer-letter prediction position rather than the start-of-solution position."
        ),
    )
    parser.add_argument(
        "--no-image-split",
        action="store_true",
        default=False,
        help=(
            "Disable processor image-splitting (~64 visual tokens vs ~320 with "
            "splitting on). MUST match the value used during training."
        ),
    )
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    data_cfg = DataConfig(
        root=args.data_root,
        image_size=args.image_size,
        include_hint=args.include_hint,
        include_lecture=args.include_lecture,
        include_metadata=args.include_metadata,
        use_cot=args.use_cot,
        use_captions=args.use_captions,
        captions_path=args.captions_path,
        disable_image_split=args.no_image_split,
    )

    captions: dict[str, str] | None = None
    if args.use_captions:
        if not args.captions_path.exists():
            msg = (
                f"--use-captions was set but the captions file {args.captions_path} "
                f"does not exist. Generate it first with scripts/generate_captions.py "
                f"or drop --use-captions."
            )
            raise FileNotFoundError(msg)
        captions = load_captions(args.captions_path)
        logger.info("Loaded %d captions from %s", len(captions), args.captions_path)

    logger.info("Loading model from %s ...", args.checkpoint)
    model, processor = load_model_and_processor(
        args.checkpoint,
        args.base_model,
        disable_image_split=args.no_image_split,
    )
    logger.info("Loading split=%s ...", args.split)
    samples = load_split(data_cfg, args.split)
    logger.info("samples=%d", len(samples))

    t0 = perf_counter()
    if args.inference_mode == "logits":
        predictions = predict_split_logits(
            model,
            processor,
            samples,
            data_cfg,
            batch_size=args.batch_size,
            prompt_style=args.prompt_style,
            save_logits_path=args.save_logits,
            captions=captions,
            logit_prefix=args.logit_prefix,
        )
    else:
        predictions = predict_split(
            model,
            processor,
            samples,
            data_cfg,
            batch_size=args.batch_size,
            prompt_style=args.prompt_style,
            use_cot=args.use_cot,
            captions=captions,
        )
    dt = perf_counter() - t0
    logger.info("Predicted %d in %.1fs (%.2f samples/s)", len(predictions), dt, len(predictions) / dt)

    # Write CSV in id-order matching sample_submission.csv
    sample_ss = pd.read_csv(args.data_root / "sample_submission.csv")
    sample_ss["answer"] = sample_ss["id"].map(predictions).fillna(0).astype(int)
    sample_ss.to_csv(args.out, index=False)
    logger.info("Wrote %s (%d rows)", args.out, len(sample_ss))

    if args.split == "val":
        # Quick val accuracy report
        gold = {s.id: s.answer_index for s in samples if s.answer_index is not None}
        correct = sum(1 for sid, pred in predictions.items() if gold.get(sid) == pred)
        logger.info("VAL accuracy: %d/%d = %.4f", correct, len(gold), correct / max(1, len(gold)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
