"""The PyTorch Lightning DataModule that ties everything in data/ together.

A DataModule's job is to be the single place that knows how to go from "a directory on
disk" to "dataloaders a Trainer can call .fit()/.validate() with". Putting that logic
here (instead of inline in a training script) is what lets the exact same data pipeline
be reused by scripts/train.py, scripts/evaluate.py, and the test suite.

Each self-supervised method needs a *different* augmentation pipeline (DINO wants
multiple global+local crops, SimCLR wants a simple pair of views, ...), so this module
also owns a small factory function, `build_ssl_transform`, that picks the right Lightly
transform for the method named in the config.
"""

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

# Methods that only need a trivial "pick the n_local_views if the transform supports
# it" style mapping; kept explicit (not clever/generic) so it's obvious at a glance
# which Lightly transform class backs which method name used throughout the configs.
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
        # DINO and DINOv2 share the same multi-crop augmentation recipe (2 global crops
        # the student+teacher both see, plus several small "local" crops only the
        # student sees). DINOv2 additionally does patch *masking* inside the model's
        # training_step, not in this transform -- see models/dinov2.py.
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
    """A plain, single-view transform used for anything that is *not* pretraining:
    the online kNN-eval callback, the final linear-probe/kNN evaluation, and the kNN
    "reference bank". No random augmentation -- we want a stable, repeatable embedding
    for a given image, not a different crop every time we look at it.
    """
    mean, std = IMAGENET_NORMALIZE["mean"], IMAGENET_NORMALIZE["std"]
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


class LithosDataModule(pl.LightningDataModule):
    """Loads LITHOS PPL patches and exposes the dataloaders training/eval code needs.

    Reads a single prepared index CSV (``patch_index.csv`` by default -- the output of
    ``scripts/prepare_data.py``, which joins the real download's two fold-index files
    with each patch's own per-patch label file; see ``src/lithos_ssl/data/splits.py``'s
    module docstring for why that join is needed and isn't done here). This class
    itself doesn't know or care about that raw, two-level layout -- it just wants
    ``Sec_patch`` / ``Binary_label`` / ``Multiclass_label`` columns.

    Four dataloaders come out of this, all built from the same section-level
    train/val/test split (see data/splits.py) so no thin section ever appears in more
    than one of them:

    - ``train_dataloader``:      SSL-augmented views, labels ignored. Used by Trainer.fit.
    - ``train_eval_dataloader``: a capped, *labelled*, non-augmented subset of the train
                                  split. This is the "reference bank" the kNN classifier
                                  searches against -- it needs real images with real
                                  labels, not augmented pretraining views.
    - ``val_dataloader``:        labelled, non-augmented. Used as the kNN "query" set by
                                  the online-eval callback during training, and for
                                  hyperparameter/checkpoint decisions.
    - ``test_dataloader``:       labelled, non-augmented, held out until the very end
                                  for the final reported numbers in RESULTS.md.
    """

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

        # Defensive filter: even though we expect the CSV to only list PPL patches
        # (per the authors' own loader), we don't actually control what the real
        # Kaggle download's CSV looks like, so we filter on the PPL marker ourselves
        # rather than trusting the file. This is the second of three independent PPL
        # guarantees in this codebase (prepare_data.py is the first, LithosPPLDataset's
        # constructor assertion is the third).
        df = df[df[PATCH_PATH_COLUMN].astype(str).str.contains(PPL_MARKER, regex=False)]
        df = section_level_split(
            df, val_fraction=self.val_fraction, test_fraction=self.test_fraction, seed=self.seed
        )

        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]

        eval_transform = build_eval_transform(self.image_size)

        # `method=None` means "I only want the labelled eval dataloaders" (this is how
        # scripts/evaluate.py uses this class -- it never trains, so it has no use for
        # an SSL multi-view transform, and building one would just mean knowing an
        # arbitrary method name that's irrelevant to what it's doing).
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
        # drop_last=True: several SSL losses (SimCLR's NT-Xent, MSN's prototype
        # assignment) compare samples *within* a batch, so a ragged final batch would
        # either crash or silently change the loss's behaviour.
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
