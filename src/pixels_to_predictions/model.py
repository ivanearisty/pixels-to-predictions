"""SmolVLM loading + LoRA attachment.

The competition pins us to ``HuggingFaceTB/SmolVLM-500M-Instruct``. We attach a
LoRA adapter to (by default) the language-model attention projections, freezing
everything else. The budget check in ``budget.py`` is the gatekeeper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import BASE_MODEL_ID
from .budget import audit_parameters

if TYPE_CHECKING:
    from torch import nn

    from .budget import ParameterAudit
    from .config import LoRAConfig, RunConfig


def load_base_model_and_processor(
    model_id: str = BASE_MODEL_ID,
    *,
    disable_image_split: bool = False,
) -> tuple[nn.Module, Any]:
    """Load the frozen SmolVLM base model and its processor from HuggingFace.

    Returns the model with all params frozen and the associated image/text
    processor for preparing inputs.

    Args:
        model_id: HuggingFace model id (pinned to SmolVLM-500M-Instruct).
        disable_image_split: When True, sets the processor's image-splitting
            flag off so each image is encoded as a single tile rather than
            split into N+1 sub-tiles. Drops visual-token count per sample
            from ~320 to ~64 (Idefics3 pixel-shuffle reduces 256 SigLIP
            patches to 64). Affects training and inference equally; pass the
            same value at predict time.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    if disable_image_split:
        processor.image_processor.do_image_splitting = False
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    )
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


def attach_lora(model: nn.Module, cfg: LoRAConfig) -> nn.Module:
    """Wrap the model with a LoRA / DoRA adapter according to ``cfg``.

    ``cfg.target_modules`` is a tuple of leaf-module name suffixes that peft
    will match against module paths. ``cfg.scope`` decides whether the vision
    encoder is excluded:
      - ``"text"`` (default): keep vision_model frozen (preferred for budget).
      - ``"all"``:           adapt vision_model too (expensive).
    """
    from peft import LoraConfig, get_peft_model

    peft_kwargs: dict[str, Any] = {
        "r": cfg.r,
        "lora_alpha": cfg.alpha,
        "lora_dropout": cfg.dropout,
        "bias": cfg.bias,
        "target_modules": list(cfg.target_modules),
        "task_type": "CAUSAL_LM",
        "use_dora": cfg.use_dora,
        "use_rslora": cfg.use_rslora,
    }
    if cfg.scope == "text":
        # Keep the vision encoder frozen even though target_modules suffixes
        # like q_proj also exist in the vision tower.
        peft_kwargs["exclude_modules"] = r".*vision_model.*"
    elif cfg.scope != "all":
        msg = f"Unknown LoRA scope: {cfg.scope!r}"
        raise ValueError(msg)

    peft_cfg = LoraConfig(**peft_kwargs)
    return get_peft_model(model, peft_cfg)


def build_model(run_cfg: RunConfig) -> tuple[nn.Module, Any, ParameterAudit]:
    """Build the SmolVLM + LoRA model for a given run config.

    Returns:
        Tuple of ``(model, processor, audit)`` where ``audit`` has already had
        :meth:`~.budget.ParameterAudit.assert_under_budget` called on it.
    """
    model, processor = load_base_model_and_processor(
        run_cfg.training.model_id,
        disable_image_split=run_cfg.data.disable_image_split,
    )
    model = attach_lora(model, run_cfg.lora)
    audit = audit_parameters(model)
    audit.assert_under_budget()
    return model, processor, audit
