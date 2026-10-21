"""lithos_ssl: self-supervised pretraining experiments on the LITHOS thin-section dataset.

This package is organised the way the experiment itself flows:

    data/    -> turn raw LITHOS image patches + CSV into PyTorch Lightning dataloaders
    models/  -> one LightningModule per self-supervised method (DINO, DINOv2, MSN, SimCLR)
                plus a frozen-backbone wrapper for evaluating pretrained DINOv3
    eval/    -> turn a trained (or frozen) backbone into downstream accuracy numbers
                (linear probe, k-nearest-neighbour classifier) and the callback that
                uses those numbers to decide which checkpoint is "best"

Nothing in this package talks to the network or to Kaggle -- that happens once, by hand,
via the instructions in the top-level README.md.
"""
