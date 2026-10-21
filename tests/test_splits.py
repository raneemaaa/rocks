"""Tests for src/lithos_ssl/data/splits.py.

Three things matter most here, and all three come straight from the user's
requirements or from what we learned authenticating against the real Kaggle dataset:
1. the raw-download-to-index join (build_patch_index) works correctly
2. the train/val/test split never leaks a thin section across splits
3. nothing in this codebase can ever pick up an XPL (cross-polarized) image
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from lithos_ssl.data.splits import (
    BINARY_LABEL_COLUMN,
    MULTICLASS_LABEL_COLUMN,
    PATCH_PATH_COLUMN,
    PPL_MARKER,
    PREPARED_INDEX_NAME,
    XPL_MARKER,
    build_patch_index,
    section_id_from_patch_path,
    section_level_split,
)


def _write_patch(root: Path, section: str, patch_id: str, grain_rows: list[dict]) -> str:
    """Write one patch's folder (images + its own center_component.csv) and return its
    relative PPL path, mirroring the real download's layout exactly."""
    patch_dir = root / "GMAS_LOTE" / section / patch_id
    patch_dir.mkdir(parents=True, exist_ok=True)
    (patch_dir / f"{PPL_MARKER}.png").write_bytes(b"not a real image, just needs to exist")
    (patch_dir / f"{XPL_MARKER}.png").write_bytes(b"not a real image, just needs to exist")
    if grain_rows:
        with open(patch_dir / "center_component.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(grain_rows[0].keys()))
            writer.writeheader()
            writer.writerows(grain_rows)
    return str(Path("GMAS_LOTE") / section / patch_id / f"{PPL_MARKER}.png")


def test_build_patch_index_joins_fold_files_with_per_patch_labels(tmp_path: Path) -> None:
    paths = [
        _write_patch(tmp_path, "s1", "0", [{MULTICLASS_LABEL_COLUMN: 3, BINARY_LABEL_COLUMN: 0}]),
        _write_patch(tmp_path, "s1", "1", [{MULTICLASS_LABEL_COLUMN: 1, BINARY_LABEL_COLUMN: 1}]),
        _write_patch(tmp_path, "s2", "0", [{MULTICLASS_LABEL_COLUMN: 2, BINARY_LABEL_COLUMN: 0}]),
    ]
    pd.DataFrame({PATCH_PATH_COLUMN: paths[:2]}).to_csv(
        tmp_path / "Fold1_complete_info_allclasses.csv", index=False
    )
    pd.DataFrame({PATCH_PATH_COLUMN: paths[2:]}).to_csv(
        tmp_path / "Fold2_complete_info_allclasses.csv", index=False
    )

    index_df = build_patch_index(tmp_path, show_progress=False)

    assert len(index_df) == 3
    assert set(index_df[PATCH_PATH_COLUMN]) == set(paths)
    row = index_df.set_index(PATCH_PATH_COLUMN).loc[paths[0]]
    assert row[MULTICLASS_LABEL_COLUMN] == 3
    assert row[BINARY_LABEL_COLUMN] == 0
    assert set(index_df["fold"]) == {1, 2}


def test_build_patch_index_drops_patches_with_no_label_file(tmp_path: Path) -> None:
    labeled = _write_patch(tmp_path, "s1", "0", [{MULTICLASS_LABEL_COLUMN: 3, BINARY_LABEL_COLUMN: 0}])
    unlabeled = _write_patch(tmp_path, "s1", "1", [])  # no grain rows written at all
    pd.DataFrame({PATCH_PATH_COLUMN: [labeled, unlabeled]}).to_csv(
        tmp_path / "Fold1_complete_info_allclasses.csv", index=False
    )
    pd.DataFrame({PATCH_PATH_COLUMN: []}).to_csv(
        tmp_path / "Fold2_complete_info_allclasses.csv", index=False
    )

    with pytest.warns(UserWarning, match="unlabeled"):
        index_df = build_patch_index(tmp_path, show_progress=False)

    assert list(index_df[PATCH_PATH_COLUMN]) == [labeled]


def test_build_patch_index_warns_and_keeps_first_row_on_multiple_grains(tmp_path: Path) -> None:
    path = _write_patch(
        tmp_path,
        "s1",
        "0",
        [
            {MULTICLASS_LABEL_COLUMN: 7, BINARY_LABEL_COLUMN: 1},
            {MULTICLASS_LABEL_COLUMN: 9, BINARY_LABEL_COLUMN: 0},
        ],
    )
    pd.DataFrame({PATCH_PATH_COLUMN: [path]}).to_csv(
        tmp_path / "Fold1_complete_info_allclasses.csv", index=False
    )
    pd.DataFrame({PATCH_PATH_COLUMN: []}).to_csv(
        tmp_path / "Fold2_complete_info_allclasses.csv", index=False
    )

    with pytest.warns(UserWarning, match="more than one grain row"):
        index_df = build_patch_index(tmp_path, show_progress=False)

    assert index_df.iloc[0][MULTICLASS_LABEL_COLUMN] == 7  # first row wins
    assert index_df.iloc[0]["num_grain_rows"] == 2  # but the anomaly is visible


def test_synthetic_dataset_index_is_ppl_only(synthetic_dataset_root: Path) -> None:
    df = pd.read_csv(synthetic_dataset_root / PREPARED_INDEX_NAME)
    # The prepared index should never list an XPL path, even though XPL files exist on
    # disk right next to the PPL ones (generated on purpose, to make sure nothing
    # silently picks them up -- see the next test).
    assert (df[PATCH_PATH_COLUMN].str.contains(PPL_MARKER)).all()
    assert not (df[PATCH_PATH_COLUMN].str.contains(XPL_MARKER)).any()


def test_xpl_files_exist_on_disk_but_are_never_referenced(synthetic_dataset_root: Path) -> None:
    xpl_files = list(synthetic_dataset_root.rglob(f"*{XPL_MARKER}*"))
    assert len(xpl_files) > 0, "test setup bug: synthetic generator should also write XPL files"

    df = pd.read_csv(synthetic_dataset_root / PREPARED_INDEX_NAME)
    referenced_paths = set(df[PATCH_PATH_COLUMN])
    for xpl_file in xpl_files:
        rel_path = str(xpl_file.relative_to(synthetic_dataset_root))
        assert rel_path not in referenced_paths


def test_section_id_is_two_levels_above_the_image_file() -> None:
    # Confirmed against the real download: <GMAS_LOTE[2]>/<section>/<patch_id>/ppl-0.png
    assert section_id_from_patch_path("GMAS_LOTE/7/132/ppl-0.png") == "GMAS_LOTE/7"
    assert section_id_from_patch_path("GMAS_LOTE2/Muestra057/70/ppl-0.png") == "GMAS_LOTE2/Muestra057"


def test_section_level_split_has_no_leakage(synthetic_dataset_root: Path) -> None:
    df = pd.read_csv(synthetic_dataset_root / PREPARED_INDEX_NAME)
    split_df = section_level_split(df, val_fraction=0.25, test_fraction=0.25, seed=0)

    split_df["section_id"] = split_df[PATCH_PATH_COLUMN].map(section_id_from_patch_path)
    sections_per_split = split_df.groupby("section_id")["split"].nunique()
    assert (sections_per_split == 1).all(), "a section appeared in more than one split"

    assert set(split_df["split"].unique()) == {"train", "val", "test"}
    assert len(split_df) == len(df)


def test_section_level_split_fractions_are_approximately_respected(synthetic_dataset_root: Path) -> None:
    df = pd.read_csv(synthetic_dataset_root / PREPARED_INDEX_NAME)
    split_df = section_level_split(df, val_fraction=0.25, test_fraction=0.25, seed=0)
    counts = split_df["split"].value_counts(normalize=True)
    # Section-level splitting can't hit the target fraction exactly (sections have
    # different patch counts), so we only check it's in the right ballpark.
    assert 0.1 < counts["val"] < 0.4
    assert 0.1 < counts["test"] < 0.4


def test_section_level_split_rejects_too_few_sections() -> None:
    df = pd.DataFrame(
        {
            PATCH_PATH_COLUMN: [
                "GMAS_LOTE/a/0/ppl-0.png",
                "GMAS_LOTE/a/1/ppl-0.png",
                "GMAS_LOTE/b/0/ppl-0.png",
            ],
            MULTICLASS_LABEL_COLUMN: [0, 1, 0],
        }
    )
    with pytest.raises(ValueError, match="at least 3 distinct sections"):
        section_level_split(df, val_fraction=0.3, test_fraction=0.3, seed=0)
