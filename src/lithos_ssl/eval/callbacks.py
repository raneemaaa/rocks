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
        pl_module.log("val_knn_acc", accuracy, prog_bar=True, logger=True)
