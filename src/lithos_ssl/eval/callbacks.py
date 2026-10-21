"""The callback that makes "save the best checkpoint" mean something during SSL training.

Here's the problem it solves: during self-supervised pretraining there are no labels,
so the only number we have to watch live is the pretext loss (DINO loss, NT-Xent,
...). But a lower pretext loss does not reliably mean better *downstream* features --
different methods' losses aren't even comparable to each other, and a method can keep
driving its own loss down for reasons that don't help classification at all. If we just
told `ModelCheckpoint` to save whichever epoch had the lowest training loss, we could
easily keep a worse checkpoint than an earlier one.

The fix (standard practice in SSL papers, including the original DINO code): periodically
pause training, freeze the current backbone, run the cheap kNN-accuracy check from
eval/knn.py using a small labelled reference/query split, and log that accuracy as a
genuine Lightning metric. `ModelCheckpoint(monitor="val/knn_acc", mode="max")` then
keeps the checkpoint from whichever epoch actually classified minerals best -- which is
the thing we actually care about for later fine-tuning.
"""

from __future__ import annotations

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from lithos_ssl.eval.embed import extract_embeddings
from lithos_ssl.eval.knn import knn_accuracy


class OnlineKNNEvalCallback(pl.Callback):
    def __init__(
        self,
        reference_dataloader: DataLoader,
        query_dataloader: DataLoader,
        k: int = 20,
        every_n_epochs: int = 1,
    ) -> None:
        super().__init__()
        self.reference_dataloader = reference_dataloader
        self.query_dataloader = query_dataloader
        self.k = k
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch_is_due = (trainer.current_epoch + 1) % self.every_n_epochs == 0
        is_last_epoch = trainer.current_epoch + 1 >= trainer.max_epochs
        if not (epoch_is_due or is_last_epoch):
            return

        reference_embeddings, reference_labels = extract_embeddings(
            pl_module, self.reference_dataloader, device=pl_module.device
        )
        query_embeddings, query_labels = extract_embeddings(
            pl_module, self.query_dataloader, device=pl_module.device
        )
        accuracy = knn_accuracy(
            reference_embeddings, reference_labels, query_embeddings, query_labels, k=self.k
        )
        # logger=True so it also shows up in TensorBoard; this is the metric
        # scripts/train.py's ModelCheckpoint callback is configured to monitor. Metric
        # name deliberately has no "/" -- Lightning's checkpoint filename templating
        # would otherwise interpret it as a subdirectory.
        pl_module.log("val_knn_acc", accuracy, prog_bar=True, logger=True)
