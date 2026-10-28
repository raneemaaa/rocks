from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_embeddings(
    model, dataloader: DataLoader, device: str | torch.device = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Run `model.embed(images)` over an entire dataloader.

    Returns ``(embeddings, labels)`` as NumPy arrays of shape ``(N, D)`` and ``(N,)``.
    Expects the dataloader to yield ``(image_tensor, label)`` pairs, which is exactly
    what :class:`lithos_ssl.data.lithos_dataset.LithosPPLDataset` with a plain
    (non-multi-view) eval transform produces.
    """
    was_training = getattr(model, "training", False)
    model.eval()

    all_embeddings = []
    all_labels = []
    for images, labels in dataloader:
        images = images.to(device)
        embeddings = model.embed(images)
        all_embeddings.append(embeddings.detach().cpu().numpy())
        all_labels.append(labels.numpy() if torch.is_tensor(labels) else np.asarray(labels))

    if was_training:
        model.train()

    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)
