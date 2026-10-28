from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import pytorch_lightning as pl
from lightly.transforms.dino_transform import DINOTransform
from lightly.transforms.msn_transform import MSNTransform
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lithos_ssl.data.lithos_dataset import LithosPPLDataset
from lithos_ssl.data.splits import (
    MULTICLASS_LABEL_COLUMN,
    PATCH_PATH_COLUMN,
    PPL_MARKER,
    PREPARED_INDEX_NAME,
    section_level_split,
)

_MULTI_CROP_METHODS = {"dino", "dinov2"}


def build_ssl_transform(method: str, image_size: int, **transform_kwargs) -> Callable:
    """Return the Lightly augmentation pipeline used *during pretraining* for `method`.

    Every one of these returns a *list* (DINO/DINOv2/MSN) or *tuple* (SimCLR) of
    augmented views of the same source image when called on a PIL image -- that's the
    whole point of these methods: compare multiple views of one image against each
    other, with no labels involved.
    """
    local_crop_size = transform_kwargs.pop("local_crop_size", max(image_size // 2, 16))
    focal_size = transform_kwargs.pop("focal_size", max(image_size // 2, 16))

    if method in _MULTI_CROP_METHODS:
        return DINOTransform(
            global_crop_size=image_size,
            local_crop_size=local_crop_size,
            **transform_kwargs,
        )
    if method == "msn":
        return MSNTransform(
            random_size=image_size,
            focal_size=focal_size,
            **transform_kwargs,
        )
    if method == "simclr":
        return SimCLRTransform(input_size=image_size, **transform_kwargs)
    raise ValueError(f"Unknown SSL method '{method}'. Expected one of: dino, dinov2, msn, simclr.")


def build_eval_transform(image_size: int) -> Callable:
    mean, std = IMAGENET_NORMALIZE["mean"], IMAGENET_NORMALIZE["std"]
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


class LithosDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str | Path,
        method: Optional[str] = None,
        csv_name: str = PREPARED_INDEX_NAME,
        image_size: int = 224,
        batch_size: int = 64,
        num_workers: int = 4,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int = 42,
        label_column: str = MULTICLASS_LABEL_COLUMN,
        knn_reference_size: int = 2000,
        transform_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.method = method
        self.csv_name = csv_name
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.label_column = label_column
        self.knn_reference_size = knn_reference_size
        self.transform_kwargs = transform_kwargs or {}

    def setup(self, stage: Optional[str] = None) -> None:
        csv_path = self.root / self.csv_name
        df = pd.read_csv(csv_path)

        df = df[df[PATCH_PATH_COLUMN].astype(str).str.contains(PPL_MARKER, regex=False)]
        df = section_level_split(
            df, val_fraction=self.val_fraction, test_fraction=self.test_fraction, seed=self.seed
        )

        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]

        eval_transform = build_eval_transform(self.image_size)

        if self.method is not None:
            ssl_transform = build_ssl_transform(self.method, self.image_size, **self.transform_kwargs)
            self.train_dataset = LithosPPLDataset(
                train_df, self.root, transform=ssl_transform, label_column=None
            )

        reference_df = train_df.sample(
            n=min(self.knn_reference_size, len(train_df)), random_state=self.seed
        )
        self.train_eval_dataset = LithosPPLDataset(
            reference_df, self.root, transform=eval_transform, label_column=self.label_column
        )
        self.val_dataset = LithosPPLDataset(
            val_df, self.root, transform=eval_transform, label_column=self.label_column
        )
        self.test_dataset = LithosPPLDataset(
            test_df, self.root, transform=eval_transform, label_column=self.label_column
        )

    def train_dataloader(self) -> DataLoader:
        if self.method is None:
            raise RuntimeError(
                "This LithosDataModule was built with method=None (eval-only mode), "
                "so it has no SSL-augmented train_dataloader. Pass a method name "
                "('dino'/'dinov2'/'msn'/'simclr') if you intend to train with it."
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def train_eval_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
