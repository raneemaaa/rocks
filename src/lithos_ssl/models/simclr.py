from __future__ import annotations

import pytorch_lightning as pl
import torch
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead

from lithos_ssl.models.backbones import build_resnet_backbone


class SimCLR(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        lr: float = 0.06,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.backbone, embed_dim = build_resnet_backbone(backbone_name)
        self.projection_head = SimCLRProjectionHead(embed_dim, hidden_dim, output_dim)
        self.criterion = NTXentLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Feature vector used by downstream evaluation (linear probe / kNN).

        Unlike DINO/DINOv2/MSN there's no separate teacher/target network here -- just
        the one backbone -- so evaluation simply runs it without the projection head
        (the projection head is only needed to compute the contrastive loss; the
        backbone's own output is the representation we actually care about).
        """
        with torch.no_grad():
            return self.backbone(x).flatten(start_dim=1)

    def training_step(self, batch, batch_idx):
        views, _labels = batch  # labels are unused: this is self-supervised pretraining
        x0, x1 = views
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
