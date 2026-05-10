"""One-sample smoke test: parse a train row, load its image, format the prompt.

Does NOT load the model — that's a separate concern. This script verifies the
data pipeline alone, which is the most common failure point on first setup.

Usage:
    python scripts/sanity_check.py --split train --limit 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pixels_to_predictions.config import DataConfig
from pixels_to_predictions.data import format_user_turn, load_image, load_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("sanity_check")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    cfg = DataConfig(root=args.data_root, max_train_samples=args.limit, max_val_samples=args.limit)
    samples = load_split(cfg, args.split)[: args.limit]
    if not samples:
        print("No samples found. Did you run scripts/setup_data.py?")
        return 1

    for s in samples:
        print("=" * 72)
        print(f"id: {s.id}   subject={s.subject!r}  topic={s.topic!r}")
        print(f"image: {s.image_path}  exists={s.image_path.exists()}")
        if s.image_path.exists():
            img = load_image(s.image_path, image_size=cfg.image_size)
            print(f"  image mode={img.mode} size={img.size}")
        print(f"answer_index: {s.answer_index}  answer_letter: {s.answer_letter}")
        print()
        print(format_user_turn(s, include_hint=cfg.include_hint, include_lecture=cfg.include_lecture))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
