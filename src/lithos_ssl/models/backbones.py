"""Shared CNN backbone builder, used by the DINO (v1) and SimCLR LightningModules.

DINOv2 and MSN need a *maskable* Vision Transformer (the model has to be able to zero
out some patch tokens before they ever reach the transformer -- a plain forward pass
can't do that), so they build their own backbone directly next to their training logic
in models/dinov2.py and models/msn.py using Lightly's dedicated masked-ViT wrapper
classes. This file only covers the simpler "ordinary CNN, no masking" case.
"""

from __future__ import annotations

import torch.nn as nn
import torchvision


def build_resnet_backbone(name: str = "resnet18") -> tuple[nn.Module, int]:
    """Build a torchvision ResNet with its classification head removed.

    Returns ``(backbone, embed_dim)`` where ``embed_dim`` is the size of the flattened
    feature vector the backbone produces per image -- every SSL method's projection
    head needs to know this number to wire itself up correctly.
    """
    builders = {
        "resnet18": (torchvision.models.resnet18, 512),
        "resnet34": (torchvision.models.resnet34, 512),
        "resnet50": (torchvision.models.resnet50, 2048),
    }
    if name not in builders:
        raise ValueError(f"Unknown backbone '{name}'. Expected one of: {sorted(builders)}.")
    build_fn, embed_dim = builders[name]

    resnet = build_fn()
    # A ResNet is Sequential(conv1, bn1, relu, maxpool, layer1..4, avgpool, fc).
    # Dropping the final `fc` layer turns it from "1000-way ImageNet classifier" into
    # a general-purpose feature extractor that outputs a pooled embed_dim-length vector.
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    return backbone, embed_dim
