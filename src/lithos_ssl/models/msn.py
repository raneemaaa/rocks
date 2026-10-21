"""MSN: "Masked Siamese Networks for Label-Efficient Learning" (Assran et al., 2022).

The core idea, in plain English
--------------------------------
Like DINO, MSN compares a "target" view of an image against "anchor" views of the same
image using a momentum-updated network, with no negative pairs. The twist: the anchor
views are heavily *masked* -- a large fraction of their patch tokens are simply dropped
before the transformer ever sees them (not blurred or blacked out: removed from the
sequence entirely, which is also why this is relatively cheap to train, since the
transformer does less work per anchor view). The target view is left intact.

Instead of comparing raw embeddings directly, both the target and the (masked) anchor
embeddings are first expressed as soft assignments to a shared bank of learned
"prototype" vectors (think: a small, trainable, non-semantic codebook). The model is
trained so that a masked anchor's prototype assignment matches its unmasked target's
assignment -- i.e. even after you remove most of its patches, the model should still be
able to tell *which* exemplar it's looking at.

Terminology note (it trips people up): here "target" (`self.backbone` /
`self.projection_head`) is the frozen, momentum-updated network -- the role DINO calls
the "teacher" -- and "anchor" (`self.anchor_backbone` / `self.anchor_projection_head`)
is the one actually trained by the optimizer -- the role DINO calls the "student". The
anchor is also the one that gets masked.

Closely follows Lightly's reference implementation
(lightly/examples/pytorch_lightning/msn.py), restructured to read its hyperparameters
(including the ViT's size -- useful for fast local smoke tests) from a Hydra config
(configs/model/msn.yaml), and with an `embed()` method added for evaluation.
"""

from __future__ import annotations

import copy

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from lightly.loss import MSNLoss
from lightly.models import utils
from lightly.models.modules import MaskedVisionTransformerTorchvision
from lightly.models.modules.heads import MSNProjectionHead


class MSN(pl.LightningModule):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 6,
        hidden_dim: int = 384,
        mlp_dim: int = 384 * 4,
        projection_hidden_dim: int = 2048,
        projection_output_dim: int = 256,
        num_prototypes: int = 1024,
        mask_ratio: float = 0.15,
        momentum: float = 0.996,
        lr: float = 1.5e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.mask_ratio = mask_ratio
        self.momentum = momentum

        vit = torchvision.models.VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
        )
        # "Target" (DINO would call this the "teacher"): frozen, EMA-updated, sees the
        # full, unmasked view.
        self.backbone = MaskedVisionTransformerTorchvision(vit=vit)
        self.projection_head = MSNProjectionHead(
            hidden_dim, projection_hidden_dim, projection_output_dim
        )

        # "Anchor" (DINO would call this the "student"): the one actually trained by
        # the optimizer; it's what sees the masked views. Copied *before* we freeze the
        # target below, so the anchor copies start out trainable (requires_grad=True).
        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)

        # The learned prototype bank every view's embedding gets soft-assigned to.
        self.prototypes = nn.Linear(projection_output_dim, num_prototypes, bias=False).weight
        self.criterion = MSNLoss()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Feature vector used by downstream evaluation (linear probe / kNN): the
        target backbone's class token on the *unmasked* image -- the stable, EMA-updated
        network, same reasoning as DINO's embed()."""
        with torch.no_grad():
            return self.backbone(images=x)

    def encode_masked(self, anchors: torch.Tensor) -> torch.Tensor:
        """Run the anchor (trainable) backbone on a batch of views with most of their
        patch tokens randomly dropped before encoding."""
        batch_size, _, _, width = anchors.shape
        seq_length = (width // self.anchor_backbone.vit.patch_size) ** 2
        idx_keep, _ = utils.random_token_mask(
            size=(batch_size, seq_length), mask_ratio=self.mask_ratio, device=self.device
        )
        out = self.anchor_backbone(images=anchors, idx_keep=idx_keep)
        return self.anchor_projection_head(out)

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.anchor_backbone, self.backbone, self.momentum)
        utils.update_momentum(self.anchor_projection_head, self.projection_head, self.momentum)

        views, _labels = batch  # labels are unused: this is self-supervised pretraining
        views = [view.to(self.device, non_blocking=True) for view in views]
        # MSNTransform's output order: 1 "target" random-crop view, 1 "anchor"
        # random-crop view, then several small "focal" crop views (all anchors too).
        targets = views[0]
        anchors = views[1]
        anchors_focal = torch.cat(views[2:], dim=0)

        targets_out = self.projection_head(self.backbone(images=targets))
        anchors_out = self.encode_masked(anchors)
        anchors_focal_out = self.encode_masked(anchors_focal)
        anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)

        loss = self.criterion(anchors_out, targets_out, self.prototypes.data)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Only the anchor network + prototypes are trained with gradients; the target
        # network is updated purely via the momentum average above.
        params = [
            *self.anchor_backbone.parameters(),
            *self.anchor_projection_head.parameters(),
            self.prototypes,
        ]
        return torch.optim.AdamW(params, lr=self.hparams.lr)
