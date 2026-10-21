"""One LightningModule per self-supervised method, plus a frozen-backbone wrapper.

`MODEL_REGISTRY` is the single place that maps the short name used throughout the
configs (``configs/model/<name>.yaml``, the ``model=<name>`` Hydra override, the
``--model`` flag on scripts/evaluate.py) to the class that implements it. Every config
field under ``configs/model/<name>.yaml`` (other than ``name`` and ``transform``, which
describe the data pipeline, not the model) is passed straight through as a constructor
keyword argument -- see scripts/train.py's `build_model`.
"""

from lithos_ssl.models.dino import DINO
from lithos_ssl.models.dinov2 import DINOv2
from lithos_ssl.models.frozen_backbone import FrozenBackbone
from lithos_ssl.models.msn import MSN
from lithos_ssl.models.simclr import SimCLR

# Trainable self-supervised methods: scripts/train.py looks a name up here.
MODEL_REGISTRY = {
    "dino": DINO,
    "dinov2": DINOv2,
    "msn": MSN,
    "simclr": SimCLR,
}

# Frozen, evaluation-only baselines: scripts/evaluate.py looks a name up here instead
# (they have no training_step/checkpoint -- see models/frozen_backbone.py).
FROZEN_MODEL_REGISTRY = {
    "dinov3_frozen": FrozenBackbone,
}

__all__ = [
    "DINO",
    "DINOv2",
    "MSN",
    "SimCLR",
    "FrozenBackbone",
    "MODEL_REGISTRY",
    "FROZEN_MODEL_REGISTRY",
]
