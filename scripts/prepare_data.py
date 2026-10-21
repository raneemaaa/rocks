"""Validate a freshly-downloaded LITHOS dataset before you spend hours training on it.

Usage:
    python scripts/prepare_data.py --root data/raw/lithos

Why this script exists: the LITHOS Kaggle page requires a logged-in account to even
browse its file listing, so this whole codebase was built against the schema
documented in the dataset authors' own public code
(github.com/BCV-Uniandes/LITHOS/data_loader.py) -- a `center_component.csv` with
`Sec_patch`/`Binary_label`/`Multiclass_label` columns and a `ppl-0`/`xpl-0` filename
marker for polarization -- rather than against the real files. Run this once, right
after you download+unzip the dataset, to catch a schema mismatch immediately with a
clear error instead of a confusing failure three hours into a training run.

Every check below points at the exact line in src/lithos_ssl/data/splits.py to fix if
your real download's layout differs from what we assumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lithos_ssl.data.splits import (  # noqa: E402
    BINARY_LABEL_COLUMN,
    MULTICLASS_LABEL_COLUMN,
    PATCH_PATH_COLUMN,
    PPL_MARKER,
    XPL_MARKER,
    section_id_from_patch_path,
    section_level_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/lithos", help="Where you unzipped the Kaggle download.")
    parser.add_argument("--csv-name", default="center_component.csv")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    csv_path = root / args.csv_name

    if not csv_path.exists():
        raise SystemExit(
            f"Could not find {csv_path}. Make sure you've downloaded and unzipped the "
            f"Kaggle dataset there (see README.md's 'Getting the data' section), or "
            f"pass --root/--csv-name to point at wherever you put it."
        )

    df = pd.read_csv(csv_path)
    required_columns = [PATCH_PATH_COLUMN, BINARY_LABEL_COLUMN, MULTICLASS_LABEL_COLUMN]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{csv_path} is missing expected column(s): {missing}.\nFound columns: "
            f"{list(df.columns)}.\nThe real download's schema differs from what this "
            f"project assumed -- update PATCH_PATH_COLUMN / BINARY_LABEL_COLUMN / "
            f"MULTICLASS_LABEL_COLUMN at the top of src/lithos_ssl/data/splits.py to "
            f"match, then re-run this script."
        )

    n_total = len(df)
    is_ppl = df[PATCH_PATH_COLUMN].astype(str).str.contains(PPL_MARKER, regex=False)
    is_xpl = df[PATCH_PATH_COLUMN].astype(str).str.contains(XPL_MARKER, regex=False)
    df_ppl = df[is_ppl].copy()

    print(f"{csv_path}")
    print(f"  total rows:                 {n_total}")
    print(f"  PPL rows ('{PPL_MARKER}'):      {int(is_ppl.sum())}  <- what this project trains on")
    print(f"  XPL rows ('{XPL_MARKER}'):      {int(is_xpl.sum())}  <- never loaded by this project")

    if is_ppl.sum() == 0:
        raise SystemExit(
            f"No row's '{PATCH_PATH_COLUMN}' path contains the marker '{PPL_MARKER}'. "
            f"The real filename convention differs from what this project assumed -- "
            f"update PPL_MARKER / XPL_MARKER at the top of src/lithos_ssl/data/splits.py "
            f"to match whatever actually distinguishes PPL from XPL filenames here."
        )

    # Spot-check that a sample of the PPL paths the CSV lists actually exist on disk --
    # catches a wrong --root or a CSV whose paths are relative to something else.
    sample = df_ppl.sample(n=min(20, len(df_ppl)), random_state=0)
    missing_files = [p for p in sample[PATCH_PATH_COLUMN] if not (root / p).exists()]
    if missing_files:
        raise SystemExit(
            f"{len(missing_files)}/{len(sample)} sampled image paths don't exist under "
            f"{root} (e.g. '{missing_files[0]}'). Check --root points at the directory "
            f"'{PATCH_PATH_COLUMN}' paths are relative to."
        )
    print(f"  spot-checked {len(sample)} PPL image files on disk: all found")

    n_sections = df_ppl[PATCH_PATH_COLUMN].map(section_id_from_patch_path).nunique()
    print(f"  distinct thin sections (best-effort heuristic): {n_sections}")
    if n_sections < 10:
        print(
            "  WARNING: that's a suspiciously small number of sections for a dataset "
            "this size. section_id_from_patch_path() in src/lithos_ssl/data/splits.py "
            "is probably not extracting the real section id correctly for this "
            "download's folder layout -- fix it there before trusting the train/val/"
            "test split (used everywhere downstream) to actually prevent patch leakage "
            "between splits."
        )

    print(f"\n  {MULTICLASS_LABEL_COLUMN} distribution (PPL rows):")
    print(df_ppl[MULTICLASS_LABEL_COLUMN].value_counts().sort_index().to_string())

    split_df = section_level_split(
        df_ppl, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed
    )
    print(
        f"\n  section-level split (val_fraction={args.val_fraction}, "
        f"test_fraction={args.test_fraction}, seed={args.seed}):"
    )
    print(split_df["split"].value_counts().to_string())

    print(
        "\nLooks good. You can now run, for example:\n"
        f"  python scripts/train.py model=simclr data.root={root}\n"
    )


if __name__ == "__main__":
    main()
