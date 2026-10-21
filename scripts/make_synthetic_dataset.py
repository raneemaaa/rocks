"""Generate a tiny, fully synthetic LITHOS-shaped dataset for a fast local smoke test.

Usage:
    python scripts/make_synthetic_dataset.py --root data/synthetic

This is *not* real data -- just random-noise JPEGs with the same CSV schema and
PPL/XPL filename convention as the real dataset (see
src/lithos_ssl/data/splits.py:generate_synthetic_lithos_dataset, which this wraps).
Its only purpose is letting you run the full training + evaluation pipeline end to end
in seconds, to confirm everything is wired together correctly, before pointing it at
the real ~211k-image download and letting it run for hours. For example:

    python scripts/make_synthetic_dataset.py --root data/synthetic
    python scripts/train.py model=simclr data.root=data/synthetic data.image_size=32 \\
        trainer.max_epochs=2
    python scripts/evaluate.py --model simclr --checkpoint checkpoints/simclr/last.ckpt \\
        --data-root data/synthetic --image-size 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lithos_ssl.data.splits import generate_synthetic_lithos_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/synthetic")
    parser.add_argument("--n-sections", type=int, default=12)
    parser.add_argument("--patches-per-section", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = generate_synthetic_lithos_dataset(
        root=Path(args.root),
        n_sections=args.n_sections,
        patches_per_section=args.patches_per_section,
        image_size=args.image_size,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    n_images = args.n_sections * args.patches_per_section
    print(f"Wrote {n_images} synthetic PPL images (+{n_images} XPL counterparts, never used) under {args.root}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
