from __future__ import annotations

import copy

import pytorch_lightning as pl
import torch
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

from lithos_ssl.models.backbones import build_resnet_backbone


class DINO(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        hidden_dim: int = 512,
        bottleneck_dim: int = 64,
        output_dim: int = 2048,
        lr: float = 1e-3,
        max_epochs: int = 10,
        momentum_start: float = 0.996,
        momentum_end: float = 1.0,
        warmup_teacher_temp_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.max_epochs = max_epochs
        self.momentum_start = momentum_start
        self.momentum_end = momentum_end

        backbone, embed_dim = build_resnet_backbone(backbone_name)

        # Student: the network we actually train with backprop.
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            embed_dim, hidden_dim, bottleneck_dim, output_dim, freeze_last_layer=1
        )

        # Teacher: an exact architectural copy of the student, but its weights are
        # *never* touched by the optimizer (deactivate_requires_grad) -- they only ever
        # move via the momentum update in training_step.
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(embed_dim, hidden_dim, bottleneck_dim, output_dim)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(
            output_dim=output_dim, warmup_teacher_temp_epochs=warmup_teacher_temp_epochs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.student_backbone(x).flatten(start_dim=1)
        return self.student_head(y)

    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        y = self.teacher_backbone(x).flatten(start_dim=1)
        return self.teacher_head(y)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Feature vector used by downstream evaluation (linear probe / kNN).

        We use the *teacher* backbone, not the student: it's the exponential moving
        average of the student over the whole run, so its features are more stable and
        it's standard DINO practice to evaluate with it.
        """
        with torch.no_grad():
            return self.teacher_backbone(x).flatten(start_dim=1)

    def training_step(self, batch, batch_idx):
        # Ramp the teacher's momentum from momentum_start towards momentum_end over
        # training: early on the teacher should adapt fairly quickly (student
        # features are still close to random init), later it should barely move so its
        # targets stay stable while the student fine-tunes against them.
        momentum = cosine_schedule(self.current_epoch, self.max_epochs, self.momentum_start, self.momentum_end)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        views, _labels = batch  # labels are unused: this is self-supervised pretraining
        views = [view.to(self.device) for view in views]
        global_views = views[:2]  # the teacher only ever sees the two global crops

        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_after_backward(self) -> None:
        # DINOProjectionHead's last layer is a weight-normalised linear layer; its
        # gradient is unstable for the first epoch, so Lightly recommends freezing it
        # briefly (`freeze_last_layer=1` above sets how many epochs this covers).
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
