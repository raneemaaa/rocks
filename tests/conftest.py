"""Shared pytest fixtures.

One synthetic dataset is generated once per test session (not once per test) and
reused everywhere -- it's randomly generated but deterministic (fixed seed), and
nothing in the test suite mutates it, so sharing it just saves a bit of time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lithos_ssl.data.splits import generate_synthetic_lithos_dataset


@pytest.fixture(scope="session")
def synthetic_dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("synthetic_lithos")
    generate_synthetic_lithos_dataset(
        root=root,
        n_sections=8,
        patches_per_section=10,
        image_size=48,
        num_classes=4,
        seed=0,
    )
    return root
