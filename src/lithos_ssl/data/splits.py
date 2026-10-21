"""Section-level train/val/test splitting, and a synthetic LITHOS-like dataset for tests.

Why "section-level" and not just a random row-level split?
------------------------------------------------------------
LITHOS patches are small crops cut out of a much smaller number of whole thin-section
slides. Patches from the *same* slide look extremely similar to each other (same rock,
same lighting, same mineral assemblage, often overlapping neighbourhoods of the same
grain). If we split patches randomly, near-duplicate patches from one slide end up on
both sides of the train/val/test boundary, the model can "cheat" by recognising the
slide rather than the mineral, and every reported accuracy number is optimistically
biased. Grouping by the parent thin section before splitting removes that leakage, so
the accuracy we measure actually reflects generalisation to *unseen* slides -- which is
the setting the eventual company fine-tuning will face too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

# Column names, as published in the LITHOS authors' own data-loading code
# (github.com/BCV-Uniandes/LITHOS/data_loader.py). We centralise them here so that if
# the real Kaggle download uses slightly different names, there is exactly one place
# to fix it (see also configs/data/lithos.yaml).
PATCH_PATH_COLUMN = "Sec_patch"
BINARY_LABEL_COLUMN = "Binary_label"
MULTICLASS_LABEL_COLUMN = "Multiclass_label"

# The authors' loader finds the cross-polarized counterpart of a PPL patch by doing a
# literal string substitution: image_path_ppl.replace("ppl-0", "xpl-0"). That tells us
# the polarization is encoded as this substring in the filename, which is what we use
# to (a) keep PPL images and (b) actively refuse to ever load an XPL image.
PPL_MARKER = "ppl-0"
XPL_MARKER = "xpl-0"


def section_id_from_patch_path(patch_path: str) -> str:
    """Best-effort extraction of the parent thin-section id from a patch path.

    We don't yet know the exact folder layout of the real Kaggle download (the dataset
    page requires a Kaggle login to browse, so we couldn't inspect it while building
    this pipeline). The LITHOS patch-extraction convention is "all patches cut from the
    same slide live under one folder, or share a filename prefix before the patch
    index". We use the *parent directory name* if the path has more than one
    component, otherwise the part of the filename before the first underscore.

    If this heuristic doesn't match the real download, override it: pass a different
    ``section_id_fn`` to :func:`section_level_split`, or fix it here once -- every
    caller in this codebase goes through this one function.
    """
    parts = Path(patch_path).parts
    if len(parts) > 1:
        return parts[-2]
    stem = Path(patch_path).stem
    return stem.split("_")[0]


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
    """Create a tiny, fully synthetic dataset with the same *shape* as LITHOS.

    This writes real (tiny) JPEG files to disk plus a ``center_component.csv`` with the
    same column names the real dataset uses, so every piece of code downstream (dataset
    class, splitting, datamodule, training loop) can be exercised end-to-end without
    the real ~211k-image download -- which requires Kaggle credentials we deliberately
    never touch. Used by the test suite and by the local smoke-training run.

    For every section we also write one XPL counterpart image per patch (never listed
    in the CSV, exactly like the real dataset) so tests can assert our pipeline never
    picks those up.

    Returns the path to the written CSV.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for section_i in range(n_sections):
        section_name = f"section{section_i:03d}"
        section_dir = images_dir / section_name
        section_dir.mkdir(parents=True, exist_ok=True)
        for patch_i in range(patches_per_section):
            # Random RGB noise is enough: we're testing plumbing, not image realism.
            ppl_array = rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
            xpl_array = rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)

            ppl_name = f"{section_name}_patch{patch_i:03d}_{PPL_MARKER}.jpg"
            xpl_name = f"{section_name}_patch{patch_i:03d}_{XPL_MARKER}.jpg"
            Image.fromarray(ppl_array).save(section_dir / ppl_name)
            Image.fromarray(xpl_array).save(section_dir / xpl_name)

            rel_path = str(Path("images") / section_name / ppl_name)
            multiclass = int(rng.integers(0, num_classes))
            rows.append(
                {
                    PATCH_PATH_COLUMN: rel_path,
                    BINARY_LABEL_COLUMN: int(multiclass == 0),
                    MULTICLASS_LABEL_COLUMN: multiclass,
                }
            )

    df = pd.DataFrame(rows)
    csv_path = root / "center_component.csv"
    df.to_csv(csv_path, index=False)
    return csv_path
