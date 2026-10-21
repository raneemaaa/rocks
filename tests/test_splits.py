"""Tests for src/lithos_ssl/data/splits.py.

Two things matter most here, and both come straight from the user's requirements:
1. the train/val/test split never leaks a thin section across splits
2. nothing in this codebase can ever pick up an XPL (cross-polarized) image
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lithos_ssl.data.splits import (
    MULTICLASS_LABEL_COLUMN,
    PATCH_PATH_COLUMN,
    PPL_MARKER,
    XPL_MARKER,
    generate_synthetic_lithos_dataset,
    section_id_from_patch_path,
    section_level_split,
)


def test_synthetic_dataset_is_ppl_only(synthetic_dataset_root: Path) -> None:
    df = pd.read_csv(synthetic_dataset_root / "center_component.csv")
    # The CSV itself should never list an XPL path, even though XPL files exist on disk
    # right next to the PPL ones (generated on purpose, to make sure nothing silently
    # picks them up -- see the next test).
    assert (df[PATCH_PATH_COLUMN].str.contains(PPL_MARKER)).all()
    assert not (df[PATCH_PATH_COLUMN].str.contains(XPL_MARKER)).any()


def test_xpl_files_exist_on_disk_but_are_never_referenced(synthetic_dataset_root: Path) -> None:
    xpl_files = list(synthetic_dataset_root.rglob(f"*{XPL_MARKER}*"))
    assert len(xpl_files) > 0, "test setup bug: synthetic generator should also write XPL files"

    df = pd.read_csv(synthetic_dataset_root / "center_component.csv")
    referenced_paths = set(df[PATCH_PATH_COLUMN])
    for xpl_file in xpl_files:
        rel_path = str(xpl_file.relative_to(synthetic_dataset_root))
        assert rel_path not in referenced_paths


def test_section_level_split_has_no_leakage(synthetic_dataset_root: Path) -> None:
    df = pd.read_csv(synthetic_dataset_root / "center_component.csv")
    split_df = section_level_split(df, val_fraction=0.25, test_fraction=0.25, seed=0)

    split_df["section_id"] = split_df[PATCH_PATH_COLUMN].map(section_id_from_patch_path)
    sections_per_split = split_df.groupby("section_id")["split"].nunique()
    assert (sections_per_split == 1).all(), "a section appeared in more than one split"

    assert set(split_df["split"].unique()) == {"train", "val", "test"}
    assert len(split_df) == len(df)


def test_section_level_split_fractions_are_approximately_respected(synthetic_dataset_root: Path) -> None:
    df = pd.read_csv(synthetic_dataset_root / "center_component.csv")
    split_df = section_level_split(df, val_fraction=0.25, test_fraction=0.25, seed=0)
    counts = split_df["split"].value_counts(normalize=True)
    # Section-level splitting can't hit the target fraction exactly (sections have
    # different patch counts), so we only check it's in the right ballpark.
    assert 0.1 < counts["val"] < 0.4
    assert 0.1 < counts["test"] < 0.4


def test_section_level_split_rejects_too_few_sections() -> None:
    df = pd.DataFrame(
        {
            PATCH_PATH_COLUMN: ["a/x_ppl-0.jpg", "a/y_ppl-0.jpg", "b/z_ppl-0.jpg"],
            MULTICLASS_LABEL_COLUMN: [0, 1, 0],
        }
    )
    with pytest.raises(ValueError, match="at least 3 distinct sections"):
        section_level_split(df, val_fraction=0.3, test_fraction=0.3, seed=0)
