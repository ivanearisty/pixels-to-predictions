"""Extract the competition zip and canonicalise the image directory layout.

The zip ships images under ``images/images/{train,val,test}/`` — a doubled
``images/`` prefix that pandas and downstream code don't expect. This script
flattens it to ``data/images/{train,val,test}/``.

Usage:
    python scripts/setup_data.py --zip ~/WorkDir/pixels-to-predictions.zip
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path

logger = logging.getLogger("p2p.setup_data")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser("setup_data")
    parser.add_argument("--zip", type=Path, required=True, help="Path to pixels-to-predictions.zip")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing data/ contents.")
    args = parser.parse_args(argv)

    if not args.zip.exists():
        logger.error("Zip not found: %s", args.zip)
        return 1

    args.data_root.mkdir(parents=True, exist_ok=True)
    if any(args.data_root.iterdir()) and not args.force:
        logger.error(
            "data_root %s is not empty -- pass --force to overwrite (destructive).",
            args.data_root,
        )
        return 2

    if args.force and args.data_root.exists():
        shutil.rmtree(args.data_root)
        args.data_root.mkdir(parents=True)

    logger.info("Extracting %s -> %s", args.zip, args.data_root)
    with zipfile.ZipFile(args.zip) as zf:
        zf.extractall(args.data_root)

    # Flatten images/images/<split> -> images/<split>
    nested = args.data_root / "images" / "images"
    if nested.exists():
        logger.info("Flattening %s -> %s", nested, args.data_root / "images")
        for sub in nested.iterdir():
            target = args.data_root / "images" / sub.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(sub), str(target))
        nested.rmdir()

    logger.info("Done. Contents:")
    for p in sorted(args.data_root.iterdir()):
        if p.is_dir():
            count = sum(1 for _ in p.rglob("*") if _.is_file())
            logger.info("  %-20s  (%d files)", p.name + "/", count)
        else:
            logger.info("  %-20s  (%d bytes)", p.name, p.stat().st_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
