from __future__ import annotations

import math
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch

from lithos_ssl.data.datamodule import LithosDataModule, build_eval_transform
from lithos_ssl.models.dino import DINO
from lithos_ssl.models.dinov2 import DINOv2
from lithos_ssl.models.msn import MSN
from lithos_ssl.models.simclr import SimCLR


def _tiny_trainer(max_epochs: int = 1) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=max_epochs,
        limit_train_batches=2,
        accelerator="cpu",  # deterministic + available everywhere CI/this machine runs
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def _assert_finite_and_fits(model: pl.LightningModule, datamodule: LithosDataModule) -> None:
    trainer = _tiny_trainer()
    trainer.fit(model, datamodule=datamodule)
    # If training produced a NaN/Inf loss, Lightning's callback_metrics still holds the
    # last value logged -- check it rather than trusting "fit() didn't raise".
    logged_losses = [v for k, v in trainer.callback_metrics.items() if "loss" in k]
    assert logged_losses, "expected at least one loss metric to have been logged"
    for value in logged_losses:
        assert math.isfinite(float(value)), f"non-finite loss logged: {value}"


def test_dino_trains(synthetic_dataset_root: Path) -> None:
    dm = LithosDataModule(
        root=synthetic_dataset_root,
        method="dino",
        image_size=32,
        batch_size=4,
        num_workers=0,
        transform_kwargs={"n_local_views": 2},
    )
    dm.setup()
    model = DINO(
        backbone_name="resnet18", hidden_dim=64, bottleneck_dim=16, output_dim=128, max_epochs=1
    )
    _assert_finite_and_fits(model, dm)


def test_dinov2_trains(synthetic_dataset_root: Path) -> None:
    # Lightly's block-masking (random_block_mask) masks at least ~10% of patches, in
    # blocks of >=4 patches each -- on a grid smaller than this, it can end up asked
    # to mask fewer patches than one block requires and raise. 128 / patch_size(16)
    # gives an 8x8 = 64-patch grid, comfortably above that floor.
    image_size = 128
    dm = LithosDataModule(
        root=synthetic_dataset_root,
        method="dinov2",
        image_size=image_size,
        batch_size=4,
        num_workers=0,
        transform_kwargs={"n_local_views": 2},
    )
    dm.setup()
    model = DINOv2(
        backbone_name="vit_tiny_patch16_224",
        image_size=image_size,
        teacher_temp_warmup_epochs=1,
    )
    _assert_finite_and_fits(model, dm)


def test_msn_trains(synthetic_dataset_root: Path) -> None:
    image_size = 32
    dm = LithosDataModule(
        root=synthetic_dataset_root,
        method="msn",
        image_size=image_size,
        batch_size=4,
        num_workers=0,
        transform_kwargs={"focal_views": 2},
    )
    dm.setup()
    model = MSN(
        image_size=image_size,
        patch_size=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=32,
        mlp_dim=64,
        projection_hidden_dim=64,
        projection_output_dim=32,
        num_prototypes=64,
    )
    _assert_finite_and_fits(model, dm)


def test_simclr_trains(synthetic_dataset_root: Path) -> None:
    dm = LithosDataModule(
        root=synthetic_dataset_root, method="simclr", image_size=32, batch_size=4, num_workers=0
    )
    dm.setup()
    model = SimCLR(backbone_name="resnet18", hidden_dim=64, output_dim=64)
    _assert_finite_and_fits(model, dm)


@pytest.mark.parametrize(
    "model_fn, image_size, expected_dim",
    [
        (lambda: DINO(backbone_name="resnet18"), 32, 512),
        (
            lambda: DINOv2(backbone_name="vit_tiny_patch16_224", image_size=64),
            64,
            192,
        ),
        (
            lambda: MSN(image_size=32, patch_size=8, num_layers=2, num_heads=2, hidden_dim=32, mlp_dim=64),
            32,
            32,
        ),
        (lambda: SimCLR(backbone_name="resnet18"), 32, 512),
    ],
)
def test_embed_produces_expected_shape(model_fn, image_size: int, expected_dim: int) -> None:
    model = model_fn()
    model.eval()
    transform = build_eval_transform(image_size)
    # A couple of plain RGB "images" run straight through model.embed(), the same
    # function the kNN callback and scripts/evaluate.py use.
    from PIL import Image

    images = torch.stack(
        [transform(Image.new("RGB", (image_size, image_size))) for _ in range(3)]
    )
    embeddings = model.embed(images)
    assert embeddings.shape == (3, expected_dim)
    assert torch.isfinite(embeddings).all()
