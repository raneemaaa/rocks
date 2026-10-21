"""Train one self-supervised model on LITHOS PPL patches.

Usage:
    python scripts/train.py                              # defaults: model=simclr
    python scripts/train.py model=dinov2
    python scripts/train.py model=msn trainer.max_epochs=50 data.batch_size=32

See configs/train.yaml for every available override, and configs/model/*.yaml for
method-specific hyperparameters. What this script actually does, end to end:

1. build a LithosDataModule (PPL-only patches, split by thin section -- see
   src/lithos_ssl/data/)
2. build the chosen LightningModule (DINO / DINOv2 / MSN / SimCLR -- see
   src/lithos_ssl/models/)
3. train it with a PyTorch Lightning Trainer, while an OnlineKNNEvalCallback
   periodically checks downstream kNN accuracy on a held-out split (see
   src/lithos_ssl/eval/callbacks.py) so ModelCheckpoint can save the epoch that was
   actually *best at the real task*, not just the one with the lowest pretext loss
4. always additionally save the final ("last") checkpoint too, for later fine-tuning

This script does *not* compute the final linear-probe/kNN numbers that go in
results/RESULTS.md -- that's scripts/evaluate.py, run afterwards against whichever
checkpoint this script produced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Lets `python scripts/train.py` work right after cloning, before anyone has run
# `pip install -e .`. The editable install (see README.md) is still recommended --
# it's what makes `import lithos_ssl` work from tests/ and from a notebook too -- but
# shouldn't be *required* just to kick off a first training run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lithos_ssl.data.datamodule import LithosDataModule  # noqa: E402
from lithos_ssl.eval.callbacks import OnlineKNNEvalCallback  # noqa: E402
from lithos_ssl.models import MODEL_REGISTRY  # noqa: E402
from lithos_ssl.utils.seed import set_seed  # noqa: E402


def build_model(model_cfg: DictConfig) -> pl.LightningModule:
    """Turn `configs/model/<name>.yaml` into an instantiated LightningModule.

    Every field in that file except ``name`` (which model to build) and ``transform``
    (data-augmentation knobs; consumed by the datamodule instead, not the model) is
    passed straight through as a constructor keyword argument. This 1:1 mapping
    between config fields and constructor arguments is deliberate: it means adding a
    new tunable hyperparameter is a one-line change in the model file's __init__ and a
    one-line addition to its YAML, with nothing else to keep in sync.
    """
    kwargs = OmegaConf.to_container(model_cfg, resolve=True)
    name = kwargs.pop("name")
    kwargs.pop("transform", None)
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(MODEL_REGISTRY)}.")
    return MODEL_REGISTRY[name](**kwargs)


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))

    datamodule = LithosDataModule(
        root=cfg.data.root,
        method=cfg.model.name,
        csv_name=cfg.data.csv_name,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_fraction=cfg.data.val_fraction,
        test_fraction=cfg.data.test_fraction,
        seed=cfg.data.seed,
        label_column=cfg.data.label_column,
        knn_reference_size=cfg.data.knn_reference_size,
        transform_kwargs=OmegaConf.to_container(cfg.model.transform, resolve=True),
    )
    # Called explicitly, rather than left for Trainer.fit to call implicitly, because
    # the kNN-eval callback below needs its reference/query dataloaders *before*
    # trainer.fit(...) is invoked.
    datamodule.setup()

    model = build_model(cfg.model)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="best",
        monitor="val_knn_acc",
        mode="max",
        save_top_k=1,
        save_last=True,  # always also write checkpoints/<method>/last.ckpt
    )
    knn_eval_callback = OnlineKNNEvalCallback(
        reference_dataloader=datamodule.train_eval_dataloader(),
        query_dataloader=datamodule.val_dataloader(),
        k=cfg.knn_eval.k,
        every_n_epochs=cfg.knn_eval.every_n_epochs,
    )
    logger = TensorBoardLogger(save_dir="results/logs", name=cfg.model.name)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=[checkpoint_callback, knn_eval_callback],
        logger=logger,
    )
    trainer.fit(model, datamodule=datamodule)

    print(f"\nBest checkpoint (val_knn_acc={checkpoint_callback.best_model_score}):")
    print(f"  {checkpoint_callback.best_model_path}")
    print(f"Last checkpoint:\n  {checkpoint_callback.last_model_path}")
    print(
        "\nNext step: "
        f"python scripts/evaluate.py --model {cfg.model.name} "
        f"--checkpoint {checkpoint_callback.best_model_path}"
    )


if __name__ == "__main__":
    main()
