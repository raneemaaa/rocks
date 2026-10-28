from __future__ import annotations

from pathlib import Path

import numpy as np
import pytorch_lightning as pl

from lithos_ssl.data.datamodule import LithosDataModule
from lithos_ssl.eval.callbacks import OnlineKNNEvalCallback
from lithos_ssl.eval.knn import knn_accuracy
from lithos_ssl.eval.linear_probe import linear_probe_accuracy
from lithos_ssl.models.simclr import SimCLR


def _make_separable_clusters(n_per_class: int = 30, n_classes: int = 4, dim: int = 8, seed: int = 0):
    # Cluster *centers* are fixed regardless of `seed` -- only the per-point noise
    # varies. That's what makes two calls with different seeds act like independent
    # reference/query draws from the *same* classes, instead of from unrelated ones.
    centers = np.random.default_rng(12345).normal(scale=10.0, size=(n_classes, dim))
    noise_rng = np.random.default_rng(seed)
    embeddings, labels = [], []
    for class_idx, center in enumerate(centers):
        embeddings.append(center + noise_rng.normal(scale=0.1, size=(n_per_class, dim)))
        labels.append(np.full(n_per_class, class_idx))
    return np.concatenate(embeddings), np.concatenate(labels)


def test_knn_accuracy_is_high_on_trivially_separable_clusters() -> None:
    ref_emb, ref_labels = _make_separable_clusters(seed=0)
    query_emb, query_labels = _make_separable_clusters(seed=1)
    accuracy = knn_accuracy(ref_emb, ref_labels, query_emb, query_labels, k=5)
    assert accuracy > 0.95


def test_linear_probe_accuracy_is_high_on_trivially_separable_clusters() -> None:
    train_emb, train_labels = _make_separable_clusters(seed=0)
    test_emb, test_labels = _make_separable_clusters(seed=1)
    accuracy = linear_probe_accuracy(train_emb, train_labels, test_emb, test_labels)
    assert accuracy > 0.95


def test_knn_accuracy_near_chance_on_random_noise() -> None:
    rng = np.random.default_rng(0)
    n_classes = 4
    ref_emb = rng.normal(size=(200, 8))
    ref_labels = rng.integers(0, n_classes, size=200)
    query_emb = rng.normal(size=(100, 8))
    query_labels = rng.integers(0, n_classes, size=100)
    accuracy = knn_accuracy(ref_emb, ref_labels, query_emb, query_labels, k=5)
    # Not an exact bound (it's still random), just a sanity check that unrelated
    # embeddings and labels don't somehow score near-perfect "accuracy".
    assert accuracy < 0.6


def test_online_knn_eval_callback_logs_a_metric_in_zero_to_one(synthetic_dataset_root: Path) -> None:
    dm = LithosDataModule(
        root=synthetic_dataset_root, method="simclr", image_size=32, batch_size=4, num_workers=0
    )
    dm.setup()
    model = SimCLR(backbone_name="resnet18", hidden_dim=32, output_dim=32)
    callback = OnlineKNNEvalCallback(
        reference_dataloader=dm.train_eval_dataloader(),
        query_dataloader=dm.val_dataloader(),
        k=5,
        every_n_epochs=1,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[callback],
    )
    trainer.fit(model, datamodule=dm)

    assert "val_knn_acc" in trainer.callback_metrics
    accuracy = float(trainer.callback_metrics["val_knn_acc"])
    assert 0.0 <= accuracy <= 1.0
