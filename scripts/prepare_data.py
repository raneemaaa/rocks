from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lithos_ssl.data.splits import (  # noqa: E402
    MULTICLASS_LABEL_COLUMN,
    PATCH_PATH_COLUMN,
    PPL_MARKER,
    PREPARED_INDEX_NAME,
    XPL_MARKER,
    build_patch_index,
    section_id_from_patch_path,
    section_level_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/lithos", help="Where you unzipped the Kaggle download.")
    parser.add_argument(
        "--out-name",
        default=PREPARED_INDEX_NAME,
        help="Filename (under --root) to write the prepared index to.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If the output index already exists, skip rebuilding it and just report on it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_path = root / args.out_name

    if not root.exists():
        raise SystemExit(
            f"{root} doesn't exist. Make sure you've downloaded and unzipped the "
            f"Kaggle dataset there (see README.md's 'Getting the data' section), or "
            f"pass --root to point at wherever you put it."
        )

    if args.skip_existing and out_path.exists():
        print(f"{out_path} already exists, skipping rebuild (--skip-existing).")
    else:
        print(f"Building {out_path} from the raw download under {root} ...")
        index_df = build_patch_index(root)
        index_df.to_csv(out_path, index=False)
        print(f"Wrote {len(index_df)} labeled patches to {out_path}")

    df = pd.read_csv(out_path)

    is_ppl = df[PATCH_PATH_COLUMN].astype(str).str.contains(PPL_MARKER, regex=False)
    is_xpl = df[PATCH_PATH_COLUMN].astype(str).str.contains(XPL_MARKER, regex=False)
    if not is_ppl.all() or is_xpl.any():
        raise SystemExit(
            f"{out_path} contains non-PPL path(s) -- this should be impossible given "
            f"how build_patch_index() constructs it. Something about the raw download "
            f"doesn't match what src/lithos_ssl/data/splits.py assumes; please report "
            f"this (do not proceed to training until it's resolved)."
        )

    # Spot-check that a sample of the indexed image files actually exist on disk.
    sample = df.sample(n=min(20, len(df)), random_state=0)
    missing_files = [p for p in sample[PATCH_PATH_COLUMN] if not (root / p).exists()]
    if missing_files:
        raise SystemExit(
            f"{len(missing_files)}/{len(sample)} sampled image paths don't exist under "
            f"{root} (e.g. '{missing_files[0]}'). Check --root points at the directory "
            f"the real download was unzipped into."
        )
    print(f"\nSpot-checked {len(sample)} PPL image files on disk: all found")

    n_sections = df[PATCH_PATH_COLUMN].map(section_id_from_patch_path).nunique()
    print(f"Distinct thin sections: {n_sections}" + (" (expect 580)" if n_sections != 580 else " -- matches the documented 580"))

    print(f"\n{MULTICLASS_LABEL_COLUMN} distribution:")
    print(df[MULTICLASS_LABEL_COLUMN].value_counts().sort_index().to_string())

    split_df = section_level_split(
        df, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed
    )
    print(
        f"\nSection-level split (val_fraction={args.val_fraction}, "
        f"test_fraction={args.test_fraction}, seed={args.seed}):"
    )
    print(split_df["split"].value_counts().to_string())

    print(
        f"\nLooks good. You can now run, for example:\n"
        f"  python scripts/train.py model=simclr data.root={root}\n"
    )


if __name__ == "__main__":
    main()
