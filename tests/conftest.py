"""Shared pytest fixtures.

One synthetic dataset is generated once per test session (not once per test) and
reused everywhere -- it's randomly generated but deterministic (fixed seed), and
nothing in the test suite mutates it, so sharing it just saves a bit of time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lithos_ssl.data.splits import (
    PREPARED_INDEX_NAME,
    build_patch_index,
    generate_synthetic_lithos_dataset,
)


@pytest.fixture(scope="session")
def synthetic_dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic dataset laid out exactly like the real Kaggle download (fold-index
    CSVs + nested per-patch folders), already run through `build_patch_index` so it has
    a ready-to-use `patch_index.csv` too -- mirrors exactly what you'd have after
    running `scripts/prepare_data.py` on the real download.
    """
    root = tmp_path_factory.mktemp("synthetic_lithos")
    generate_synthetic_lithos_dataset(
        root=root,
        n_sections=8,
        patches_per_section=10,
        image_size=48,
        num_classes=4,
        seed=0,
    )
    index_df = build_patch_index(root, show_progress=False)
    index_df.to_csv(root / PREPARED_INDEX_NAME, index=False)
    return root
