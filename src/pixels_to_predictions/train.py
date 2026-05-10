"""Train SmolVLM-500M-Instruct + LoRA on the FashionMCQ data.

The training target is a **single answer letter** per sample (A/B/C/...).
Loss is masked so only the answer tokens contribute -- the chat-templated
prompt tokens are set to -100 in the labels.

Subprocess-friendly: emits ``metrics.json`` to ``cfg.run_dir`` so the
overnight optimizer can read final accuracy / loss back.

Usage:
    python -m pixels_to_predictions.train --run-name baseline
    python -m pixels_to_predictions.train --config-json '{"training": {...}, ...}'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch

from .budget import audit_parameters
from .config import (
    DataConfig,
    GenerationConfig,
    LoRAConfig,
    RunConfig,
    TrainingConfig,
)
from .data import (
    MCQSample,
    format_assistant_turn,
    format_user_turn_with_caption,
    load_captions,
    load_image,
    load_split,
)
from .model import build_model
from .seed import seed_everything

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger("p2p.train")

USER_HEADER_FORMAT = "{question}"  # the formatting is done by data.format_user_turn


@dataclass
class CollatorBatch:
    """Container for a collated multimodal batch."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    pixel_attention_mask: torch.Tensor | None
    labels: torch.Tensor


def build_messages(sample: MCQSample, data_cfg: DataConfig, *, with_answer: bool) -> list[dict[str, Any]]:
    """Compose the chat-message list for one MCQ sample.

    Thin wrapper around :func:`build_messages_with_caption` for backward
    compatibility — equivalent to passing ``caption=None``.
    """
    return build_messages_with_caption(sample, data_cfg, with_answer=with_answer, caption=None)


def build_messages_with_caption(
    sample: MCQSample,
    data_cfg: DataConfig,
    *,
    with_answer: bool,
    caption: str | None,
) -> list[dict[str, Any]]:
    r"""Compose the chat-message list for one MCQ sample, optionally with a caption.

    The ``user`` turn carries ``[image, text]``; the optional ``assistant`` turn
    is just the gold answer letter. Metadata (subject/topic) is appended to the
    user text when ``data_cfg.include_metadata=True``. When ``caption`` is
    non-empty, ``"[Image content: {caption}]\n\n"`` is prepended to the user
    text (before any subject metadata).
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

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    if with_answer and sample.answer_letter is not None:
        assistant_text = format_assistant_turn(sample, cot=data_cfg.use_cot)
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        )
    return messages


class MCQDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
    """Index over MCQSample list. Lazy image loading happens in the collator.

    When ``data_cfg.use_captions`` is set, captions are loaded once at
    construction time from ``data_cfg.captions_path`` and stashed on
    ``self.captions`` for the collator to consume. A missing file is treated
    as "no captions" (empty dict).
    """

    def __init__(self, samples: list[MCQSample], data_cfg: DataConfig) -> None:
        """Wrap a parsed list of MCQ samples for HF Trainer consumption."""
        self.samples = samples
        self.data_cfg = data_cfg
        self.captions: dict[str, str] = (
            load_captions(data_cfg.captions_path) if data_cfg.use_captions else {}
        )
        if data_cfg.use_captions:
            logger.info(
                "Loaded %d captions from %s", len(self.captions), data_cfg.captions_path,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MCQSample:
        return self.samples[idx]


class MCQCollator:
    """Custom collator: chat-templates messages, loads images, masks prompt labels."""

    def __init__(
        self,
        processor: Any,
        data_cfg: DataConfig,
        *,
        max_length: int,
        captions: dict[str, str] | None = None,
    ) -> None:
        """Save references to the processor, data config, and optional captions."""
        self.processor = processor
        self.data_cfg = data_cfg
        self.max_length = max_length
        self.captions: dict[str, str] = captions or {}

    def __call__(self, samples: list[MCQSample]) -> dict[str, torch.Tensor]:
        prompt_texts: list[str] = []
        full_texts: list[str] = []
        images: list[list[Image]] = []  # per-sample list of images
        for s in samples:
            caption = self.captions.get(s.id) if self.captions else None
            msgs_full = build_messages_with_caption(
                s, self.data_cfg, with_answer=True, caption=caption,
            )
            msgs_prompt = build_messages_with_caption(
                s, self.data_cfg, with_answer=False, caption=caption,
            )
            prompt_texts.append(
                self.processor.apply_chat_template(msgs_prompt, add_generation_prompt=True),
            )
            full_texts.append(
                self.processor.apply_chat_template(msgs_full, add_generation_prompt=False),
            )
            images.append([load_image(s.image_path, self.data_cfg.image_size)])

        # Encode the full sequence; this gives us input_ids + pixel_values.
        full = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        # Encode prompt-only to find where to start the loss mask.
        prompt = self.processor.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        # Build labels: copy of input_ids, with prompt prefix -> -100 and pad -> -100.
        labels = full["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        for i in range(labels.size(0)):
            prompt_len = prompt["input_ids"][i].size(0)
            labels[i, :prompt_len] = -100
            if pad_id is not None:
                labels[i][full["input_ids"][i] == pad_id] = -100

        out: dict[str, torch.Tensor] = dict(full)  # input_ids, attention_mask, pixel_values, ...
        out["labels"] = labels
        return out


def _trainer_args(cfg: TrainingConfig, *, eval_enabled: bool) -> Any:
    """Build a transformers TrainingArguments for our config."""
    from transformers import TrainingArguments

    bf16 = cfg.mixed_precision == "bf16"
    fp16 = cfg.mixed_precision == "fp16"

    return TrainingArguments(
        output_dir=str(cfg.output_root / cfg.run_name),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=bf16,
        fp16=fp16,
        optim=cfg.optim,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=cfg.eval_steps if eval_enabled else None,
        save_total_limit=cfg.save_total_limit,
        max_steps=cfg.max_steps if cfg.max_steps is not None else -1,
        seed=cfg.seed,
        report_to="none",  # offline competition
        remove_unused_columns=False,  # we pass MCQSample, not column dict
        dataloader_num_workers=0,
        save_safetensors=True,
    )


def _load_config_from_json(text: str) -> RunConfig:
    """Rebuild a RunConfig from a JSON string."""
    d = json.loads(text)
    training = TrainingConfig(
        **{**d["training"], "output_root": Path(d["training"]["output_root"])},
    )
    lora = LoRAConfig(**d["lora"])
    data_d = dict(d["data"])
    data_d["root"] = Path(data_d["root"])
    if "captions_path" in data_d:
        data_d["captions_path"] = Path(data_d["captions_path"])
    data = DataConfig(**data_d)
    generation = GenerationConfig(**d["generation"])
    return RunConfig(training=training, lora=lora, data=data, generation=generation)


def _parse_cli(argv: list[str] | None = None) -> RunConfig:
    """Map argparse CLI to a RunConfig."""
    parser = argparse.ArgumentParser("pixels_to_predictions.train")
    parser.add_argument("--config-json", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="baseline")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-targets", type=str, default="q_proj,v_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--use-dora", action="store_true", default=True)
    parser.add_argument("--no-dora", dest="use_dora", action="store_false")
    parser.add_argument("--use-rslora", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument(
        "--use-cot",
        action="store_true",
        default=False,
        help=(
            "Train with chain-of-thought distillation: assistant target becomes "
            "'{solution}\\n\\nThe answer is {letter}.' instead of the bare letter. "
            "Consider raising --max-seq-length-equivalent (TrainingConfig.max_seq_length, "
            "e.g. 8192) since CoT solutions can exceed the default 4096 budget."
        ),
    )
    parser.add_argument(
        "--use-captions",
        action="store_true",
        default=False,
        help=(
            "Prepend a SmolVLM-self-generated image caption to the user turn. "
            "Captions are read from --captions-path; missing file -> no captions."
        ),
    )
    parser.add_argument(
        "--captions-path",
        type=Path,
        default=Path("data/captions.json"),
        help="JSON file mapping sample-id -> caption (built by scripts/generate_captions.py).",
    )
    parser.add_argument(
        "--no-image-split",
        action="store_true",
        default=False,
        help=(
            "Disable the processor's image-splitting step so each image is "
            "encoded as a single tile (~64 visual tokens) rather than being "
            "split into N+1 sub-tiles (~320 visual tokens). Pass the same flag "
            "at predict time."
        ),
    )
    args = parser.parse_args(argv)

    if args.config_json is not None:
        return _load_config_from_json(args.config_json)

    # CoT distillation needs much more sequence room: 600 image-patch tokens +
    # question + lecture + ~856-char solution overflows the 4096 default.
    max_seq_length = 8192 if args.use_cot else TrainingConfig.max_seq_length

    return RunConfig(
        training=TrainingConfig(
            run_name=args.run_name,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.max_steps,
            seed=args.seed,
            output_root=args.output_root,
            max_seq_length=max_seq_length,
        ),
        lora=LoRAConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            target_modules=tuple(args.lora_targets.split(",")),
            use_dora=args.use_dora,
            use_rslora=args.use_rslora,
        ),
        data=DataConfig(
            root=args.data_root,
            image_size=args.image_size,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            use_cot=args.use_cot,
            use_captions=args.use_captions,
            captions_path=args.captions_path,
            disable_image_split=args.no_image_split,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Train end-to-end. Returns shell exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = _parse_cli(argv)
    seed_everything(cfg.training.seed)

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    logger.info("Run dir: %s", cfg.run_dir)

    t0 = perf_counter()

    # 1. Build model + processor; budget audit happens inside build_model.
    logger.info("Loading SmolVLM and attaching LoRA...")
    model, processor, audit = build_model(cfg)
    logger.info("Model ready:\n%s", audit.summary())
    model.to("cuda")

    # 2. Build datasets
    logger.info("Loading training data...")
    train_samples = load_split(cfg.data, "train")
    val_samples = load_split(cfg.data, "val") if not _no_eval(argv) else []
    logger.info("train=%d  val=%d", len(train_samples), len(val_samples))

    train_ds = MCQDataset(train_samples, cfg.data)
    val_ds = MCQDataset(val_samples, cfg.data) if val_samples else None

    collator = MCQCollator(
        processor,
        cfg.data,
        max_length=cfg.training.max_seq_length,
        captions=train_ds.captions if cfg.data.use_captions else None,
    )
    eval_enabled = val_ds is not None

    # 3. Run trainer
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=_trainer_args(cfg.training, eval_enabled=eval_enabled),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    logger.info("Starting trainer.train()...")
    train_t0 = perf_counter()
    train_result = trainer.train()
    train_wall = perf_counter() - train_t0
    logger.info("Training finished in %.1fs", train_wall)

    # 4. Save final adapter
    final_dir = cfg.run_dir / "checkpoint-final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    logger.info("Saved final checkpoint to %s", final_dir)

    # 5. Re-audit (DoRA + lora may register differently after training) and dump metrics
    final_audit = audit_parameters(model)
    metrics = {
        "status": "completed",
        "run_dir": str(cfg.run_dir),
        "trainable_params": final_audit.trainable,
        "param_budget": final_audit.budget,
        "under_budget": final_audit.under_budget,
        "train_loss": float(train_result.training_loss) if train_result.training_loss is not None else None,
        "train_runtime_s": train_result.metrics.get("train_runtime") if train_result.metrics else None,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "wall_clock_s": perf_counter() - t0,
        "train_wall_clock_s": train_wall,
        "final_checkpoint": str(final_dir),
    }
    (cfg.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("metrics: %s", json.dumps(metrics, indent=2))
    return 0


def _no_eval(argv: list[str] | None) -> bool:
    """Cheap re-parse of just the --no-eval flag."""
    return bool(argv and "--no-eval" in argv)


if __name__ == "__main__":
    sys.exit(main())
