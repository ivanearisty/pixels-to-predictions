"""Dataclass configs for training, LoRA, data, and generation.

Mirrors the svg-gen convention: one frozen dataclass per concern, composed into
a top-level RunConfig. Configs are JSON-serialisable so they can be passed to
subprocess trials by the overnight optimizer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import BASE_MODEL_ID


@dataclass(frozen=True)
class DataConfig:
    """Where the competition data lives on disk and how to slice it."""

    root: Path = Path("data")
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"
    test_csv: str = "test.csv"
    sample_submission: str = "sample_submission.csv"
    # images/train, images/val, images/test under root/
    image_subdir: str = "images"
    # Optional: cap training examples for quick iteration or ASHA early-stop rungs
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    # Prompt formatting
    answer_style: str = "letter"  # "letter" -> "A"/"B"/... ; "index" -> "0"/"1"/...
    include_lecture: bool = True
    include_hint: bool = True
    # Image resizing before the vision encoder. SmolVLM default is 384; bigger
    # helps for image-heavy questions (timelines, maps with small text labels).
    image_size: int = 384
    # Add subject/topic metadata to the prompt (instructor strategy hint #6).
    include_metadata: bool = True
    # Chain-of-thought distillation: when True, the assistant target becomes
    # "{solution}\n\nThe answer is {letter}." instead of the bare letter. The
    # CSV ``solution`` column (median ~350 chars, p95 ~856 chars) supplies the
    # reasoning. Train + val have it; test does not. Bumping ``max_seq_length``
    # in TrainingConfig (e.g. 8192) is recommended when this flag is on, since
    # 600 image-patch tokens + lecture + question + 856-char solution can exceed
    # the default 4096 cap.
    use_cot: bool = False
    # Self-generated captions: when True, prepend a SmolVLM-produced
    # "[Image content: {caption}]" line to the user turn. Captions live in a
    # JSON file mapping sample-id -> caption (built by
    # ``scripts/generate_captions.py``). If the file is missing, the loader
    # returns an empty dict and per-sample formatting silently falls back to
    # the regular un-captioned prompt.
    use_captions: bool = False
    captions_path: Path = Path("data/captions.json")
    # If True, set processor.image_processor.do_image_splitting=False so each
    # image is encoded as a single tile (~64 visual tokens) instead of being
    # split into multiple sub-tiles (~320 visual tokens). Pass the same value
    # at training and inference; mismatch produces OOD inputs.
    disable_image_split: bool = False


# Common LoRA target presets. A "preset" is just a tuple of suffix strings that
# peft will match against module paths.
ATTN_TARGETS: tuple[str, ...] = ("q_proj", "v_proj")
ATTN_FULL_TARGETS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
ATTN_MLP_TARGETS: tuple[str, ...] = (
    "q_proj", "v_proj", "gate_proj", "up_proj", "down_proj",
)


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA adapter shape + target modules.

    Trainable parameter count is dominated by target_modules x num_layers x r x 2.
    The audit in ``budget.py`` will refuse any config that exceeds 5 M params,
    so it's safe to experiment freely with bigger ranks / wider targets.
    """

    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    bias: str = "none"  # "none" | "all" | "lora_only"
    # Default targets attention + MLP (instructor strategy hint #1 + #4).
    # The 5M budget at r=8 with these targets lands around ~2.2M -- safe.
    target_modules: tuple[str, ...] = ATTN_MLP_TARGETS
    # "text": adapt only the language-model layers (vision encoder stays frozen).
    # "all":  adapt the vision encoder too -- expensive, rarely worth the budget.
    scope: str = "text"  # "text" | "all"
    # DoRA (Liu et al. 2024) -- magnitude-direction decomposition, ~1% extra
    # trainable params for documented quality gains on VLM benchmarks.
    use_dora: bool = True
    # rsLoRA (Kalajdzievski 2023) -- replaces alpha/r scaling with alpha/sqrt(r)
    # so updates don't shrink to nothing at higher ranks. Free flag, ~0 overhead;
    # most useful when r >= 32. Compatible with use_dora.
    use_rslora: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Core training hyperparameters consumed by trl SFTTrainer."""

    model_id: str = BASE_MODEL_ID
    num_train_epochs: float = 2.0
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # "bf16" | "fp16" | "no"
    # Image patches eat ~600+ tokens at 384px; leave room for question + hint + lecture
    max_seq_length: int = 4096
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 25
    save_total_limit: int = 3
    seed: int = 42
    # Hard stops for debugging / ASHA rungs
    max_steps: int | None = None
    # Where to drop checkpoints / logs for this run
    run_name: str = "baseline"
    output_root: Path = Path("outputs")


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding params for inference / submission."""

    max_new_tokens: int = 4  # answer is a single token, plus some slack
    temperature: float = 0.0  # deterministic for MCQ
    top_p: float = 1.0
    do_sample: bool = False
    # When the model emits a full sentence instead of a bare letter, we fall back
    # to the first char matching the valid-choice regex.
    fallback_parse: bool = True


@dataclass(frozen=True)
class RunConfig:
    """Top-level config composed for a single training or eval run."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        out = asdict(self)
        # Path objects -> str for JSON compatibility
        out["training"]["output_root"] = str(self.training.output_root)
        out["data"]["root"] = str(self.data.root)
        out["data"]["captions_path"] = str(self.data.captions_path)
        return out

    @property
    def run_dir(self) -> Path:
        """Directory where this run's checkpoints + logs live."""
        return self.training.output_root / self.training.run_name
