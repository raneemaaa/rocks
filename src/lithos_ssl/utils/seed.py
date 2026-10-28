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
