"""DINOv2: "Learning Robust Visual Features without Supervision" (Oquab et al., 2023).

The core idea, in plain English
--------------------------------
DINOv2 is DINO (see models/dino.py -- read that one first, this builds on it) plus two
extra ingredients:

1. **iBOT patch-level loss.** DINO only ever compares the *whole-image* summary (the
   "class token") between student and teacher. DINOv2 additionally hides ("masks") a
   random block of the student's input patches and asks it to predict, for each hidden
   patch, what the teacher saw there -- a masked-image-modelling objective borrowed from
   iBOT, layered on top of DINO's image-level objective. This pushes the model to learn
   useful *local* / fine-grained features, not just a global summary -- plausibly
   valuable here, since telling two minerals apart is often about a small grain's
   texture, not the whole patch.
2. **KoLeo regularizer.** A small extra loss term that spreads the student's class-token
   embeddings out to be more uniformly distributed within a batch, which empirically
   makes nearest-neighbour-style downstream use (exactly what our kNN evaluation does)
   work better.

Both the DINO loss and the iBOT loss still use the same student/teacher-with-momentum
setup as plain DINO; nothing about *that* mechanism changes.

Because DINOv2 needs to mask individual patch *tokens* before the transformer ever sees
them (not just mask pixels in the input image), the backbone has to be a ViT wrapped in
Lightly's `MaskedVisionTransformerTIMM`, which exposes an `encode(x, mask=...)` API --
plain `backbone(x)` won't do.

Closely follows Lightly's reference implementation
(lightly/examples/pytorch_lightning/dinov2.py), restructured to read its hyperparameters
(including which ViT size to use -- useful for keeping local smoke tests fast) from a
Hydra config (configs/model/dinov2.yaml), and with an `embed()` method added for
evaluation.
"""

from __future__ import annotations

import copy
from functools import partial

import pytorch_lightning as pl
import timm
import torch
from torch import Tensor
from torch.nn import Module

from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.modules import DINOv2ProjectionHead, MaskedVisionTransformerTIMM
from lightly.models.utils import random_block_mask, update_drop_path_rate, update_momentum
from lightly.utils.scheduler import cosine_schedule, linear_warmup_schedule


def _freeze(module: Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


class DINOv2Head(Module):
    """Bundles the DINO (class-token) head and the iBOT (patch-token) head together
    so student/teacher momentum updates can treat "the head" as one object, the same
    way they treat "the backbone" as one object.
    """

    def __init__(self, dino_head: DINOv2ProjectionHead, ibot_head: DINOv2ProjectionHead) -> None:
        super().__init__()
        self.dino_head = dino_head
        self.ibot_head = ibot_head


class DINOv2(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str = "vit_small_patch16_224",
        image_size: int = 224,
        ibot_separate_head: bool = False,
        drop_path_rate: float = 0.1,
        lr: float = 1e-3,
        koleo_weight: float = 0.1,
        momentum_start: float = 0.992,
        momentum_end: float = 1.0,
        teacher_temp_start: float = 0.04,
        teacher_temp_end: float = 0.07,
        teacher_temp_warmup_epochs: int = 30,
        weight_decay_start: float = 0.04,
        weight_decay_end: float = 0.4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.koleo_weight = koleo_weight
        self.momentum_start = momentum_start
        self.momentum_end = momentum_end
        self.teacher_temp_start = teacher_temp_start
        self.teacher_temp_end = teacher_temp_end
        self.teacher_temp_warmup_epochs = teacher_temp_warmup_epochs
        self.weight_decay_start = weight_decay_start
        self.weight_decay_end = weight_decay_end

        # `img_size=image_size` matters: it fixes the ViT's internal patch grid to
        # match the crop size our data pipeline actually produces (configs/data/*.yaml),
        # which the masking logic below relies on being correct. `dynamic_img_size=True`
        # additionally lets the *local* (smaller) crops pass through the same model.
        vit_teacher = timm.create_model(
            backbone_name,
            img_size=image_size,
            pos_embed="learn",
            dynamic_img_size=True,
            init_values=1e-5,
        )
        embed_dim = getattr(vit_teacher, "embed_dim", None) or vit_teacher.num_features

        self.teacher_backbone = MaskedVisionTransformerTIMM(
            vit=vit_teacher, antialias=False, pos_embed_initialization="skip"
        )
        self.student_backbone = copy.deepcopy(self.teacher_backbone)
        # Stochastic depth (randomly dropping whole transformer blocks during training)
        # regularizes the student; the teacher doesn't need it since it's never trained
        # directly, only EMA-updated.
        update_drop_path_rate(self.student_backbone.vit, drop_path_rate=drop_path_rate, mode="uniform")
        _freeze(self.teacher_backbone)

        dino_head_fn = partial(DINOv2ProjectionHead, input_dim=embed_dim)
        teacher_dino_head, student_dino_head = dino_head_fn(), dino_head_fn()
        if ibot_separate_head:
            ibot_head_fn = partial(DINOv2ProjectionHead, input_dim=embed_dim)
            teacher_ibot_head, student_ibot_head = ibot_head_fn(), ibot_head_fn()
        else:
            # Sharing one head for both objectives is the DINOv2 paper's default and
            # uses fewer parameters; set ibot_separate_head=true in the config to split
            # them if you want to experiment with that.
            teacher_ibot_head, student_ibot_head = teacher_dino_head, student_dino_head

        self.teacher_head = DINOv2Head(dino_head=teacher_dino_head, ibot_head=teacher_ibot_head)
        self.student_head = DINOv2Head(dino_head=student_dino_head, ibot_head=student_ibot_head)
        _freeze(self.teacher_head)

        self.dino_criterion = DINOLoss()
        self.ibot_criterion = IBOTPatchLoss()
        self.koleo_criterion = KoLeoLoss()

    def forward_teacher(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.teacher_backbone.encode(x)
        cls_token = features[:, 0]  # token 0 is the whole-image summary ("[CLS]")
        return cls_token, features

    def forward_student(self, x: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor | None]:
        features = self.student_backbone.encode(x, mask=mask)
        cls_token = features[:, 0]
        masked_features = None if mask is None else features[mask]
        return cls_token, masked_features

    def embed(self, x: Tensor) -> Tensor:
        """Feature vector used by downstream evaluation (linear probe / kNN): the
        teacher's class token, unmasked -- same reasoning as plain DINO's embed()."""
        with torch.no_grad():
            cls_token, _ = self.forward_teacher(x)
        return cls_token

    def training_step(self, batch, batch_idx: int) -> Tensor:
        views, _labels = batch  # labels are unused: this is self-supervised pretraining
        views = [view.to(self.device) for view in views]
        global_views = torch.cat(views[:2])  # the 2 full-size crops (student AND teacher see these)
        local_views = torch.cat(views[2:])  # several small crops (student only)

        # --- Build the patch mask for the student's global views ---------------------
        # We hide a random contiguous block of patches (not scattered individual
        # patches -- blocks are harder to "cheat" by interpolating neighbours) from the
        # student, never from the class token (index 0, left unmasked below).
        batch_size = len(global_views)
        sequence_length = self.teacher_backbone.sequence_length
        mask = global_views.new_zeros((batch_size, sequence_length), dtype=torch.bool)
        grid_h, grid_w = self.teacher_backbone.vit.patch_embed.grid_size
        assert grid_h * grid_w == sequence_length - 1, (
            f"ViT patch grid ({grid_h}x{grid_w}) doesn't match sequence_length "
            f"({sequence_length}). This usually means configs/data/*.yaml's image_size "
            f"doesn't match configs/model/dinov2.yaml's image_size -- they must agree."
        )
        block_mask = random_block_mask(size=(batch_size, grid_h, grid_w), device=mask.device)
        mask[:, 1:] = block_mask.flatten(start_dim=1)

        # --- Teacher forward (no grad: the teacher is only ever EMA-updated) ---------
        with torch.no_grad():
            teacher_cls_token, teacher_features = self.forward_teacher(global_views)
            teacher_cls_out = self.teacher_head.dino_head(teacher_cls_token)
            teacher_masked_out = self.teacher_head.ibot_head(teacher_features[mask])

        # --- Student forward -----------------------------------------------------------
        student_global_cls, student_global_masked = self.forward_student(global_views, mask=mask)
        student_global_cls_out = self.student_head.dino_head(student_global_cls)
        student_global_masked_out = self.student_head.ibot_head(student_global_masked)

        student_local_cls, _ = self.forward_student(local_views, mask=None)
        student_local_cls_out = self.student_head.dino_head(student_local_cls)
        student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out])

        # Teacher "temperature" controls how sharply-peaked the teacher's target
        # distribution is; starting low and warming up avoids an early collapse where
        # an undertrained teacher gives overconfident (and wrong) targets.
        teacher_temp = linear_warmup_schedule(
            step=self.trainer.global_step,
            warmup_steps=int(
                self.teacher_temp_warmup_epochs
                / self.trainer.max_epochs
                * self.trainer.estimated_stepping_batches
            ),
            start_value=self.teacher_temp_start,
            end_value=self.teacher_temp_end,
        )
        dino_loss = self.dino_criterion(
            teacher_out=teacher_cls_out.chunk(2),
            student_out=student_cls_out.chunk(len(views)),
            teacher_temp=teacher_temp,
        )
        ibot_loss = self.ibot_criterion(
            teacher_out=teacher_masked_out,
            student_out=student_global_masked_out,
            mask=block_mask,
            teacher_temp=teacher_temp,
        )
        koleo_loss = self.koleo_weight * sum(
            self.koleo_criterion(t) for t in student_global_cls.chunk(2)
        )
        loss = dino_loss + ibot_loss + koleo_loss
        self.log_dict(
            {
                "train/loss": loss,
                "train/dino_loss": dino_loss,
                "train/ibot_loss": ibot_loss,
                "train/koleo_loss": koleo_loss,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log("train/loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def on_before_optimizer_step(self, optimizer, *args) -> None:
        # The last layer's learning rate is kept at 0 for the first epoch (same
        # stabilisation trick plain DINO uses via freeze_last_layer, applied here
        # directly to the optimizer since DINOv2ProjectionHead doesn't take that arg).
        if self.current_epoch < 1:
            for group in optimizer.param_groups:
                if "last_layer" in group:
                    group["lr"] = 0.0

        weight_decay = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=self.weight_decay_start,
            end_value=self.weight_decay_end,
        )
        for group in optimizer.param_groups:
            if group["weight_decay"] != 0.0:
                group["weight_decay"] = weight_decay

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=self.momentum_start,
            end_value=self.momentum_end,
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
