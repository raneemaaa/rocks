from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit


PATCH_PATH_COLUMN = "Sec_patch"
BINARY_LABEL_COLUMN = "Binary_label"
MULTICLASS_LABEL_COLUMN = "Multiclass_label"

PPL_MARKER = "ppl-0"
XPL_MARKER = "xpl-0"


FOLD_CSV_NAMES = ("Fold1_complete_info_allclasses.csv", "Fold2_complete_info_allclasses.csv")

PER_PATCH_LABEL_CSV_NAME = "center_component.csv"

PREPARED_INDEX_NAME = "patch_index.csv"


def section_id_from_patch_path(patch_path: str) -> str:
    return str(Path(patch_path).parent.parent)


def build_patch_index(
    root: Path,
    fold_csv_names: tuple[str, ...] = FOLD_CSV_NAMES,
    per_patch_label_csv_name: str = PER_PATCH_LABEL_CSV_NAME,
    show_progress: bool = True,
) -> pd.DataFrame:
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
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    fold_rows: list[list[str]] = [[], []]
    for section_i in range(n_sections):
        section_name = f"section{section_i:03d}"
        for patch_i in range(patches_per_section):
            patch_dir = root / "GMAS_LOTE" / section_name / str(patch_i)
            patch_dir.mkdir(parents=True, exist_ok=True)

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
