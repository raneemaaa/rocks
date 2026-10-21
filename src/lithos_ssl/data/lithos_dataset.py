"""The PyTorch Dataset for LITHOS patches -- PPL (plane-polarized light) only.

LITHOS pairs every plane-polarized (PPL) patch with a cross-polarized (XPL) patch of
the same spot on the slide. The user explicitly asked to train on PPL only, so this
file is written to make an XPL image structurally impossible to load: we never read
the XPL-related columns the authors' loader uses, we never apply the "ppl-0" ->
"xpl-0" substitution they use to *find* the XPL pair, and the constructor refuses to
start if it is ever handed a row whose path doesn't contain the PPL marker. There is
no flag that turns XPL on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from lithos_ssl.data.splits import (
    MULTICLASS_LABEL_COLUMN,
    PATCH_PATH_COLUMN,
    PPL_MARKER,
    XPL_MARKER,
)


class LithosPPLDataset(Dataset):
    """One row of the prepared patch index (``patch_index.csv``, built by
    ``scripts/prepare_data.py`` -- see ``data/splits.py``'s module docstring for why
    that preparation step exists) -> one PPL image (+ optional label).

    Parameters
    ----------
    df:
        A dataframe with at least the ``Sec_patch`` column (relative path to the PPL
        patch). Typically a ``train``/``val``/``test`` slice produced by
        :func:`lithos_ssl.data.splits.section_level_split`.
    root:
        Directory that ``Sec_patch`` paths are relative to.
    transform:
        Applied to the loaded PIL image. During *pretraining* this is one of Lightly's
        multi-view transforms (e.g. ``DINOTransform``), which returns a *list* of
        augmented crops instead of a single tensor -- the self-supervised loss needs
        several differently-augmented views of the same image to compare against each
        other. During *evaluation* (linear probe, kNN, the online-eval callback) it's a
        plain resize+normalize transform returning a single tensor.
    label_column:
        Which label column to return (``Multiclass_label`` by default). Pass ``None``
        if you don't have/want labels (pretraining never looks at the label anyway).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root: Path,
        transform: Optional[Callable] = None,
        label_column: Optional[str] = MULTICLASS_LABEL_COLUMN,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.label_column = label_column

        bad_rows = ~df[PATCH_PATH_COLUMN].astype(str).str.contains(PPL_MARKER, regex=False)
        if bad_rows.any():
            offending = df.loc[bad_rows, PATCH_PATH_COLUMN].head(3).tolist()
            raise ValueError(
                f"Refusing to build a dataset containing non-PPL paths (this project "
                f"trains on PPL only). First offending path(s): {offending}. Expected "
                f"every path to contain the marker '{PPL_MARKER}'."
            )
        contains_xpl = df[PATCH_PATH_COLUMN].astype(str).str.contains(XPL_MARKER, regex=False)
        assert not contains_xpl.any(), "An XPL path slipped into a PPL-only dataset."

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.root / row[PATCH_PATH_COLUMN]
        image = Image.open(image_path).convert("RGB")

        view = self.transform(image) if self.transform is not None else image

        # -1 is a harmless placeholder: pretraining never reads the label, so any
        # downstream code that *does* need real labels (eval/knn.py, linear_probe.py)
        # is expected to construct this dataset with a real `label_column`.
        label = int(row[self.label_column]) if self.label_column is not None else -1
        return view, label
