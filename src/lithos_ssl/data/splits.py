"""Turning the real LITHOS download into one clean (patch, label) index, plus
section-level train/val/test splitting, and a synthetic LITHOS-like raw dataset for tests.

The real download's layout (confirmed directly against the Kaggle dataset with
`kaggle datasets files` / sampled downloads -- see the project history for how; the
dataset page itself requires a login to browse, so this took actually authenticating
and probing it) is *two-level*, not the single flat CSV this project originally assumed
before that was possible:

1. Two "fold" index files at the dataset root, ``Fold1_complete_info_allclasses.csv``
   and ``Fold2_complete_info_allclasses.csv``. Each has exactly one column,
   ``Sec_patch``, listing PPL patch paths like ``GMAS_LOTE/7/132/ppl-0.png``. The two
   files are disjoint (a 2-fold cross-validation split) and together list all 105,802
   labeled patches -- confirmed by downloading both and checking
   ``len(fold1_paths | fold2_paths) == 105802``, which matches the dataset's own
   documented "105,802 expert-annotated grains" figure exactly. (The dataset's
   "211,604 patches" figure is these same 105,802 locations counted twice -- once per
   polarization, PPL + XPL.)
2. Every one of those patches has its *own* small ``center_component.csv`` sitting in
   its own folder (e.g. ``GMAS_LOTE/7/132/center_component.csv``), with the grain-level
   label (``Multiclass_label``, ``Binary_label``) and geometry for that patch. Since
   105,802 labeled patches exactly equals the dataset's "105,802 expert-annotated
   grains", each of these per-patch files is expected to hold exactly one grain row --
   :func:`build_patch_index` below still defensively handles 0 or >1 rows (by warning
   and taking the first row) in case a real download ever disagrees with that.

:func:`build_patch_index` (used by ``scripts/prepare_data.py``) does the one-time join
of (1) and (2) into a single ``patch_index.csv`` with ``Sec_patch`` /
``Binary_label`` / ``Multiclass_label`` columns -- which is the clean, single-file shape
every other piece of this codebase (``LithosDataModule``, ``LithosPPLDataset``,
``section_level_split`` below) actually consumes. That separation -- "messy real
layout" handled in one place, "clean index" consumed everywhere else -- is what let the
rest of this codebase stay simple and testable before the real download was ever
available to look at.

Why split by *section*, not randomly by patch?
------------------------------------------------
LITHOS patches are small crops cut from a much smaller number (580, confirmed -- see
below) of whole thin-section slides. Patches from the *same* slide look extremely
similar to each other (same rock, same lighting, same mineral assemblage, often
overlapping neighbourhoods of the same grain). If we split patches randomly,
near-duplicate patches from one slide end up on both sides of the train/val/test
boundary, the model can "cheat" by recognising the slide rather than the mineral, and
every reported accuracy number is optimistically biased. Grouping by the parent thin
section before splitting removes that leakage, so the accuracy we measure actually
reflects generalisation to *unseen* slides -- which is the setting the eventual company
fine-tuning will face too.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

# Column names, confirmed against the real Kaggle download (see module docstring).
# Centralised here so that if a future dataset version renames something, there is
# exactly one place to fix it.
PATCH_PATH_COLUMN = "Sec_patch"
BINARY_LABEL_COLUMN = "Binary_label"
MULTICLASS_LABEL_COLUMN = "Multiclass_label"

# The dataset finds the cross-polarized counterpart of a PPL patch with a literal
# string substitution on the filename ("ppl-0.png" -> "xpl-0.png"). That's what we use
# to (a) keep PPL images and (b) actively refuse to ever load an XPL image.
PPL_MARKER = "ppl-0"
XPL_MARKER = "xpl-0"

# The two fold-index files at the dataset root (see module docstring, point 1).
FOLD_CSV_NAMES = ("Fold1_complete_info_allclasses.csv", "Fold2_complete_info_allclasses.csv")
# The per-patch grain-annotation file sitting next to every patch's images (point 2).
PER_PATCH_LABEL_CSV_NAME = "center_component.csv"
# What build_patch_index's output is conventionally named once written to disk.
PREPARED_INDEX_NAME = "patch_index.csv"


def section_id_from_patch_path(patch_path: str) -> str:
    """Extract the parent thin-section id from a patch path.

    Confirmed against the real download: every patch path has the shape
    ``<GMAS_LOTE or GMAS_LOTE2>/<section>/<patch_id>/ppl-0.png`` (constant depth of 4
    path components), so "two directories up from the image file" is the thin-section
    id. We checked this gives exactly 580 distinct values across the full labeled set
    -- matching the dataset's own documented "580 thin sections" exactly -- so this
    isn't a guess.
    """
    return str(Path(patch_path).parent.parent)


def build_patch_index(
    root: Path,
    fold_csv_names: tuple[str, ...] = FOLD_CSV_NAMES,
    per_patch_label_csv_name: str = PER_PATCH_LABEL_CSV_NAME,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Join the real download's fold-index files with each patch's own label file.

    Returns a DataFrame with one row per labeled patch:
    ``Sec_patch`` (PPL path), ``Binary_label``, ``Multiclass_label``, ``fold`` (which
    fold file it came from, 1-indexed), and ``num_grain_rows`` (how many rows its own
    label CSV actually had -- 1 for essentially every patch, per the module docstring;
    kept around so a surprising download can be noticed instead of silently mislabeled).

    This does one small file read per patch (~105,802 of them on the real dataset), so
    it's meant to be run once, locally, after the full dataset is downloaded -- see
    ``scripts/prepare_data.py``, which wraps this and writes the result to
    ``<root>/patch_index.csv``.
    """
    root = Path(root)

    fold_frames = []
    for fold_number, csv_name in enumerate(fold_csv_names, start=1):
        fold_path = root / csv_name
        if not fold_path.exists():
            raise FileNotFoundError(
                f"Expected fold index file not found: {fold_path}. Make sure you "
                f"downloaded+unzipped the full Kaggle dataset at --root."
            )
        fold_df = pd.read_csv(fold_path)
        fold_df["fold"] = fold_number
        fold_frames.append(fold_df)
    index_df = pd.concat(fold_frames, ignore_index=True)

    iterator = index_df[PATCH_PATH_COLUMN]
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="Reading per-patch grain labels", unit="patch")

    binary_labels, multiclass_labels, num_grain_rows = [], [], []
    n_missing = 0
    for patch_path in iterator:
        label_path = root / Path(patch_path).parent / per_patch_label_csv_name
        try:
            with open(label_path, newline="") as f:
                rows = list(csv.DictReader(f))
        except FileNotFoundError:
            n_missing += 1
            binary_labels.append(None)
            multiclass_labels.append(None)
            num_grain_rows.append(0)
            continue

        num_grain_rows.append(len(rows))
        if not rows:
            binary_labels.append(None)
            multiclass_labels.append(None)
            continue
        # Expected case (see module docstring): exactly one grain row. If a real
        # download ever has more, we don't silently average/vote over possibly
        # different classes -- we take the first and let `num_grain_rows` in the
        # output flag that it happened, so it's visible in prepare_data.py's summary.
        first = rows[0]
        binary_labels.append(first.get(BINARY_LABEL_COLUMN))
        multiclass_labels.append(first.get(MULTICLASS_LABEL_COLUMN))

    index_df[BINARY_LABEL_COLUMN] = pd.to_numeric(pd.Series(binary_labels), errors="coerce")
    index_df[MULTICLASS_LABEL_COLUMN] = pd.to_numeric(pd.Series(multiclass_labels), errors="coerce")
    index_df["num_grain_rows"] = num_grain_rows

    if n_missing:
        warnings.warn(
            f"{n_missing}/{len(index_df)} patches listed in {fold_csv_names} had no "
            f"'{per_patch_label_csv_name}' on disk and were left unlabeled (dropped by "
            f"the caller if it filters to labeled rows)."
        )

    n_multi = int((index_df["num_grain_rows"] > 1).sum())
    if n_multi:
        warnings.warn(
            f"{n_multi} patch(es) had more than one grain row in their "
            f"'{per_patch_label_csv_name}' -- only the first row's label was kept for "
            f"each. This was expected to never happen (see module docstring); inspect "
            f"`num_grain_rows` in the returned index if you need to investigate."
        )

    before = len(index_df)
    index_df = index_df.dropna(subset=[MULTICLASS_LABEL_COLUMN]).reset_index(drop=True)
    index_df[MULTICLASS_LABEL_COLUMN] = index_df[MULTICLASS_LABEL_COLUMN].astype(int)
    index_df[BINARY_LABEL_COLUMN] = index_df[BINARY_LABEL_COLUMN].astype(int)
    if len(index_df) < before:
        warnings.warn(f"Dropped {before - len(index_df)} unlabeled patch(es) from the index.")

    return index_df


def section_level_split(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    section_id_fn=section_id_from_patch_path,
) -> pd.DataFrame:
    """Add a ``split`` column (values: "train", "val", "test") grouped by section.

    Uses sklearn's GroupShuffleSplit so that every row sharing a section id ends up in
    the same split. Splitting happens in two steps: carve off "test" first, then carve
    "val" out of what remains, so the fractions are of the *whole* dataset as given.
    """
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("val_fraction and test_fraction must be in (0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")

    groups = df[PATCH_PATH_COLUMN].map(section_id_fn)
    n_groups = groups.nunique()
    if n_groups < 3:
        raise ValueError(
            f"Need at least 3 distinct sections to form train/val/test splits, found "
            f"{n_groups}. Check that `section_id_fn` is extracting sensible ids from "
            f"'{PATCH_PATH_COLUMN}' -- see section_id_from_patch_path()."
        )

    df = df.reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(df, groups=groups))

    # val_fraction was expressed as a fraction of the *whole* dataset; convert it to a
    # fraction of the remaining train+val rows for the second split.
    val_fraction_of_remainder = val_fraction / (1.0 - test_fraction)
    inner_groups = groups.iloc[train_val_idx]
    inner_splitter = GroupShuffleSplit(
        n_splits=1, test_size=val_fraction_of_remainder, random_state=seed
    )
    train_idx_rel, val_idx_rel = next(
        inner_splitter.split(df.iloc[train_val_idx], groups=inner_groups)
    )
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    split = pd.Series("train", index=df.index, dtype=object)
    split.iloc[val_idx] = "val"
    split.iloc[test_idx] = "test"
    return df.assign(split=split)


def generate_synthetic_lithos_dataset(
    root: Path,
    n_sections: int = 6,
    patches_per_section: int = 12,
    image_size: int = 32,
    num_classes: int = 5,
    seed: int = 0,
) -> Path:
    """Create a tiny, fully synthetic dataset with the same *raw* shape as the real
    LITHOS download: two fold-index CSVs at the root, and one nested
    ``<section>/<patch_id>/`` folder per patch containing ``ppl-0.png``, ``xpl-0.png``,
    and its own ``center_component.csv`` grain label. Nothing here is real data --
    just random-noise images with the right shape -- so the test suite (and
    scripts/make_synthetic_dataset.py, for a local smoke-training run) can exercise
    the *entire* real pipeline, including the fold-join step in
    :func:`build_patch_index`, without the real ~18GB Kaggle download.

    Every patch also gets an XPL counterpart image, present on disk but never listed
    in any CSV -- exactly like the real dataset -- so tests can assert our pipeline
    never picks those up.

    Returns ``root``.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    fold_rows: list[list[str]] = [[], []]
    for section_i in range(n_sections):
        section_name = f"section{section_i:03d}"
        for patch_i in range(patches_per_section):
            patch_dir = root / "GMAS_LOTE" / section_name / str(patch_i)
            patch_dir.mkdir(parents=True, exist_ok=True)

            # Random RGB noise is enough: we're testing plumbing, not image realism.
            ppl_array = rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
            xpl_array = rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
            Image.fromarray(ppl_array).save(patch_dir / f"{PPL_MARKER}.png")
            Image.fromarray(xpl_array).save(patch_dir / f"{XPL_MARKER}.png")

            multiclass = int(rng.integers(0, num_classes))
            with open(patch_dir / PER_PATCH_LABEL_CSV_NAME, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Multiclass_label_name",
                        MULTICLASS_LABEL_COLUMN,
                        "Binary_label_name",
                        BINARY_LABEL_COLUMN,
                        "major-axis",
                        "minor-axis",
                        "x_annot",
                        "y_annot",
                    ]
                )
                writer.writerow(
                    [
                        f"class{multiclass}",
                        multiclass,
                        "pos" if multiclass == 0 else "neg",
                        int(multiclass == 0),
                        1.0,
                        0.5,
                        100.0,
                        200.0,
                    ]
                )

            rel_path = str(Path("GMAS_LOTE") / section_name / str(patch_i) / f"{PPL_MARKER}.png")
            fold_rows[patch_i % 2].append(rel_path)

    for fold_number, csv_name in enumerate(FOLD_CSV_NAMES, start=1):
        pd.DataFrame({PATCH_PATH_COLUMN: fold_rows[fold_number - 1]}).to_csv(
            root / csv_name, index=False
        )

    return root
