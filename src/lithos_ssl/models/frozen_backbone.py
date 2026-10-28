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
