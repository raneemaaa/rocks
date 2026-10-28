from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lithos_ssl.data.splits import (  # noqa: E402
    PREPARED_INDEX_NAME,
    build_patch_index,
    generate_synthetic_lithos_dataset,
)


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
    root = Path(args.root)
    generate_synthetic_lithos_dataset(
        root=root,
        n_sections=args.n_sections,
        patches_per_section=args.patches_per_section,
        image_size=args.image_size,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    n_images = args.n_sections * args.patches_per_section
    print(f"Wrote {n_images} synthetic PPL images (+{n_images} XPL counterparts, never used) under {root}")

    index_df = build_patch_index(root, show_progress=False)
    index_path = root / PREPARED_INDEX_NAME
    index_df.to_csv(index_path, index=False)
    print(f"Built prepared index: {index_path} ({len(index_df)} labeled patches)")


if __name__ == "__main__":
    main()
