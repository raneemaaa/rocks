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

        label = int(row[self.label_column]) if self.label_column is not None else -1
        return view, label
