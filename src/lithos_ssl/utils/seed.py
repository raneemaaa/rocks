"""Reproducibility helper.

Self-supervised runs involve a lot of randomness (augmentations, crop sampling,
weight init, shuffling, masking). Fixing every RNG we can reach makes two runs with
the same config comparable, and makes a failing test reproducible instead of flaky.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python's, NumPy's and PyTorch's random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # also seeds CUDA/MPS generators if a GPU backend is active
    torch.cuda.manual_seed_all(seed)
