"""A frozen, pretrained backbone used only for *evaluation* -- never trained here.

This is how DINOv3 is covered in this project. Lightly does not implement DINOv3
training (confirmed against its docs/examples before writing this codebase), and
nobody pretrains a model of that scale from scratch on a ~200k-image domain dataset
anyway -- DINOv3's whole selling point is that its off-the-shelf features are already
excellent without any fine-tuning. So instead of a training LightningModule, this file
just loads Meta's released weights and reports how good they are at telling our
mineral classes apart out of the box, as a reference point for the methods we *do*
train (DINO, DINOv2, MSN, SimCLR -- see dino.py, dinov2.py, msn.py, simclr.py).

Any Hugging Face vision model that returns a pooled embedding (DINOv3, DINOv2-via-HF,
a plain ViT, ...) works here, not just DINOv3 -- `checkpoint_id` is a config value, not
a hardcoded string.

Gating note: Meta's DINOv3 checkpoints on Hugging Face are access-gated -- you must be
logged into a Hugging Face account that has accepted the DINOv3 license before
`from_pretrained` below will succeed. Run `huggingface-cli login` (or set the
`HF_TOKEN` environment variable) yourself; this project never touches your
credentials, and the first run will simply raise a clear error if you haven't done
this yet.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class FrozenBackbone(nn.Module):
    """Wraps a pretrained Hugging Face vision model behind the same ``embed()``
    interface every trainable LightningModule in this project exposes, so
    scripts/evaluate.py can treat a from-scratch-trained model and a frozen pretrained
    one identically.

    Note on preprocessing: like every other model in this project, this expects input
    tensors already resized and ImageNet-normalized by
    :func:`lithos_ssl.data.datamodule.build_eval_transform` -- which happens to be the
    same normalization the DINO family's own Hugging Face image processors use, so
    this is consistent with how the model was actually trained. If you swap in a
    checkpoint with different expected preprocessing, use its own
    ``AutoImageProcessor`` instead and adapt this class accordingly.
    """

    def __init__(self, checkpoint_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m") -> None:
        super().__init__()
        self.checkpoint_id = checkpoint_id
        self.model = AutoModel.from_pretrained(checkpoint_id)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        # Most HF vision backbones (including the DINO family) expose a ready-pooled
        # [CLS]-token embedding here; fall back to manually indexing out the class
        # token from the raw sequence for models that don't.
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]
