"""Tests for src/lithos_ssl/data/datamodule.py and lithos_dataset.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lithos_ssl.data.datamodule import LithosDataModule
from lithos_ssl.data.splits import PATCH_PATH_COLUMN


@pytest.mark.parametrize("method", ["dino", "dinov2", "msn", "simclr"])
def test_train_dataloader_yields_multiple_views_and_no_labels(
    synthetic_dataset_root: Path, method: str
) -> None:
    dm = LithosDataModule(
        root=synthetic_dataset_root, method=method, image_size=32, batch_size=4, num_workers=0
    )
    dm.setup()
    views, labels = next(iter(dm.train_dataloader()))

    # Every SSL method's transform returns several augmented views per image (a list
    # for DINO/DINOv2/MSN, a 2-tuple for SimCLR) -- that's the whole mechanism these
    # methods rely on (see each models/*.py module docstring).
    assert isinstance(views, (list, tuple))
    assert len(views) >= 2
    for view in views:
        assert view.shape[0] == 4  # batch size
        assert view.shape[1] == 3  # RGB

    # Pretraining never looks at labels -- the datamodule still returns a batch of
    # them (for a uniform (views, label) shape), but they're the -1 placeholder.
    assert torch.equal(labels, torch.full((4,), -1))


def test_eval_dataloaders_yield_single_view_and_real_labels(synthetic_dataset_root: Path) -> None:
    dm = LithosDataModule(
        root=synthetic_dataset_root, method="simclr", image_size=32, batch_size=4, num_workers=0
    )
    dm.setup()

    for loader in (dm.train_eval_dataloader(), dm.val_dataloader(), dm.test_dataloader()):
        images, labels = next(iter(loader))
        assert images.ndim == 4  # (batch, channel, height, width) -- a single view, not a list
        assert images.shape[1:] == (3, 32, 32)
        assert labels.min() >= 0  # real labels this time, not the -1 pretraining placeholder


def test_splits_dont_overlap(synthetic_dataset_root: Path) -> None:
    dm = LithosDataModule(root=synthetic_dataset_root, method="simclr", image_size=32, num_workers=0)
    dm.setup()
    train_paths = set(dm.train_dataset.df[PATCH_PATH_COLUMN])
    val_paths = set(dm.val_dataset.df[PATCH_PATH_COLUMN])
    test_paths = set(dm.test_dataset.df[PATCH_PATH_COLUMN])
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)


def test_eval_only_mode_has_no_train_dataloader(synthetic_dataset_root: Path) -> None:
    # scripts/evaluate.py never trains, so it builds the datamodule with method=None.
    dm = LithosDataModule(root=synthetic_dataset_root, method=None, image_size=32, num_workers=0)
    dm.setup()
    with pytest.raises(RuntimeError, match="method=None"):
        dm.train_dataloader()
    # But the labelled eval dataloaders, which are all evaluate.py actually needs,
    # still work fine.
    images, labels = next(iter(dm.test_dataloader()))
    assert images.ndim == 4
