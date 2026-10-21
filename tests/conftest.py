"""Shared fixtures: a tiny synthetic chest X-ray dataset on disk."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

CLASSES = ["NORMAL", "PNEUMONIA"]


def _write_images(folder, count, rng):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        pixels = rng.integers(0, 256, size=(40, 48), dtype=np.uint8)  # grayscale, non-square
        Image.fromarray(pixels, mode="L").save(folder / f"img_{i}.jpeg")


@pytest.fixture
def fake_dataset(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "chest_xray"
    counts = {"train": (10, 30), "val": (2, 2), "test": (4, 6)}
    for split, (n_normal, n_pneu) in counts.items():
        _write_images(root / split / "NORMAL", n_normal, rng)
        _write_images(root / split / "PNEUMONIA", n_pneu, rng)
    return root
