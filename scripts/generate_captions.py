"""Pre-extract captions / OCR-style descriptions for every image with frozen SmolVLM.

The competition rules forbid external pretrained models, but the same
``HuggingFaceTB/SmolVLM-500M-Instruct`` base we fine-tune on is allowed for
preprocessing. We run it FROZEN (no LoRA, no training) over every image in
``data/images/{train,val,test}/`` with a captioning + OCR prompt and dump the
results to ``data/captions.json`` keyed by sample id. The captions can then be
prepended to the question prompt at training / inference time, especially for
diagrams, timelines, and maps where small text labels are otherwise lost.

Saves incrementally every 100 samples so a crash mid-run doesn't lose work.

Usage:
    python scripts/generate_captions.py
    python scripts/generate_captions.py --limit 2  # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch

from pixels_to_predictions import BASE_MODEL_ID
from pixels_to_predictions.data import load_image

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger("p2p.generate_captions")


CAPTION_PROMPT = (
    "Describe everything you can see in this image, especially any text labels, "
    "numbers, axis labels, captions, or diagram elements. Be concise but precise. "
    "Include any visible text verbatim."
)
SPLITS: tuple[str, ...] = ("train", "val", "test")
CHECKPOINT_EVERY = 100


def load_model_and_processor(model_id: str) -> tuple[Any, Any]:
    """Load a frozen SmolVLM base model + processor on cuda for batched generation.

    Sets ``padding_side='left'`` on the tokenizer: decoder-only models generate
    from the rightmost non-pad token, so right-padding silently corrupts the
    generation context for shorter prompts in a batch.

    Args:
        model_id: HuggingFace model id; pinned to the competition base model.

    Returns:
        A tuple of ``(model, processor)``. The model is on cuda in bf16 with
        ``requires_grad`` disabled on every parameter and ``.eval()`` set.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    )
    for p in model.parameters():
        p.requires_grad_(False)
    model.to("cuda").eval()
    return model, processor


def _build_messages() -> list[dict[str, Any]]:
    """User-turn messages for one captioning request (one image + prompt)."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        },
    ]


def discover_images(data_root: Path, image_subdir: str = "images") -> list[tuple[str, Path]]:
    """Walk ``data_root/image_subdir/{train,val,test}/`` and return (sample_id, path) tuples.

    Sample id is the filename stem (``{id}.png`` -> ``{id}``). Sorted by
    (split, id) so the iteration order is reproducible.

    Args:
        data_root: dataset root, e.g. ``data/``.
        image_subdir: subdirectory containing the per-split image folders.

    Returns:
        List of ``(sample_id, absolute_path)`` tuples covering all splits.
    """
    out: list[tuple[str, Path]] = []
    for split in SPLITS:
        split_dir = data_root / image_subdir / split
        if not split_dir.is_dir():
            logger.warning("Split directory missing: %s", split_dir)
            continue
        files = sorted(split_dir.glob("*.png"))
        logger.info("Discovered %d images in %s", len(files), split_dir)
        out.extend((p.stem, p) for p in files)
    return out


def _save_captions(captions: dict[str, str], out_path: Path) -> None:
    """Atomically write captions JSON via a tmp-file rename."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(captions, fh, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(out_path)


def _format_eta(done: int, total: int, elapsed_s: float) -> str:
    """Return ``'(elapsed=12.3s, eta=4m02s, rate=2.1/s)'`` style progress string."""
    rate = done / elapsed_s if elapsed_s > 0 else 0.0
    remaining = max(0, total - done)
    eta_s = remaining / rate if rate > 0 else 0.0
    eta_m, eta_sec = divmod(int(eta_s), 60)
    eta_h, eta_m = divmod(eta_m, 60)
    eta = f"{eta_h:d}h{eta_m:02d}m{eta_sec:02d}s" if eta_h else f"{eta_m:d}m{eta_sec:02d}s"
    return f"(elapsed={elapsed_s:.1f}s, eta={eta}, rate={rate:.2f}/s)"


@torch.no_grad()
def generate_captions(
    model: Any,
    processor: Any,
    items: list[tuple[str, Path]],
    out_path: Path,
    *,
    batch_size: int = 4,
    image_size: int = 384,
    max_new_tokens: int = 160,
) -> dict[str, str]:
    """Run frozen SmolVLM over every image and collect captions.

    Saves incrementally to ``out_path`` every ``CHECKPOINT_EVERY`` samples so
    a mid-run crash doesn't lose work. Logs progress every 50 samples with a
    timestamp + ETA. Prints the first 3 captions verbatim for sanity-checking.

    Args:
        model: loaded SmolVLM model on cuda in eval mode.
        processor: matching AutoProcessor (tokenizer must be left-padding).
        items: ``[(sample_id, image_path), ...]`` to process.
        out_path: where to write the JSON ``{sample_id: caption}`` mapping.
        batch_size: how many images per generation call. 4 fits comfortably
            on a single Ampere/Hopper GPU at 384px and 160 new tokens.
        image_size: longest-edge target for image preprocessing (matches
            :func:`pixels_to_predictions.data.load_image`).
        max_new_tokens: greedy generation budget per caption.

    Returns:
        Final ``{sample_id: caption}`` mapping. Also written to ``out_path``.
    """
    captions: dict[str, str] = {}
    total = len(items)
    if total == 0:
        logger.warning("No images to process.")
        _save_captions(captions, out_path)
        return captions

    pad_token_id = processor.tokenizer.pad_token_id

    # Pre-render the prompt text (no image yet) once -- it's identical per sample.
    prompt_text: str = processor.apply_chat_template(
        _build_messages(),
        add_generation_prompt=True,
    )

    t0 = perf_counter()
    printed_examples = 0
    last_log = 0
    for start in range(0, total, batch_size):
        batch = items[start : start + batch_size]
        ids = [sid for sid, _ in batch]
        images = [[load_image(path, image_size)] for _, path in batch]
        texts = [prompt_text for _ in batch]

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to("cuda")

        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_token_id,
        )
        prompt_len = inputs["input_ids"].size(1)
        new_ids = gen_ids[:, prompt_len:]
        decoded = processor.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        decoded = [d.strip() for d in decoded]

        for sid, caption in zip(ids, decoded, strict=True):
            captions[sid] = caption
            if printed_examples < 3:
                printed_examples += 1
                print(f"--- example {printed_examples}: {sid} ---")
                print(caption)
                print()

        done = len(captions)
        if done - last_log >= 50 or done == total:
            elapsed = perf_counter() - t0
            ts = datetime.now(UTC).isoformat(timespec="seconds")
            logger.info("[%s] %d/%d %s", ts, done, total, _format_eta(done, total, elapsed))
            last_log = done

        if done % CHECKPOINT_EVERY == 0 and done > 0:
            _save_captions(captions, out_path)
            logger.info("Checkpointed %d captions -> %s", done, out_path)

    _save_captions(captions, out_path)
    elapsed = perf_counter() - t0
    logger.info("Done: %d captions in %.1fs (%.2f/s) -> %s", len(captions), elapsed,
                len(captions) / max(elapsed, 1e-9), out_path)
    return captions


def _slice_limit(items: Iterable[tuple[str, Path]], limit: int | None) -> list[tuple[str, Path]]:
    """Apply an optional ``--limit`` cap and return a fresh list."""
    items_list = list(items)
    if limit is not None:
        return items_list[:limit]
    return items_list


def main(argv: list[str] | None = None) -> int:
    """Caption every image under ``data/images/{train,val,test}/`` to JSON."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser("generate_captions")
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                        help="Dataset root (must contain images/{train,val,test}/).")
    parser.add_argument("--out", type=Path, default=Path("data/captions.json"),
                        help="Output JSON path.")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL_ID,
                        help="HuggingFace model id (frozen base model).")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--limit", type=int, default=None,
                        help="If set, stop after N samples (smoke-test mode).")
    args = parser.parse_args(argv)

    items = _slice_limit(discover_images(args.data_root), args.limit)
    if not items:
        logger.error("No images found under %s. Did you run scripts/setup_data.py?", args.data_root)
        return 1
    logger.info("Will caption %d images. Loading frozen model %s ...", len(items), args.base_model)

    model, processor = load_model_and_processor(args.base_model)
    generate_captions(
        model,
        processor,
        items,
        args.out,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_new_tokens=args.max_new_tokens,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
