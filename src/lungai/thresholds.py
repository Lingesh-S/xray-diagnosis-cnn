"""Saving and loading a model's tuned decision threshold.

Kept free of TensorFlow imports so the web app can read it before loading a model.
"""

from __future__ import annotations

import json
from pathlib import Path


def threshold_path(model_path: str | Path) -> Path:
    """Where the tuned threshold for a model is stored (``<model>_threshold.json``)."""
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}_threshold.json")


def save_threshold(model_path: str | Path, threshold: float) -> None:
    with open(threshold_path(model_path), "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold}, f, indent=2)


def load_threshold(model_path: str | Path, default: float = 0.5) -> float:
    """Tuned threshold saved by ``lungai.evaluate --tune-threshold``, else ``default``."""
    path = threshold_path(model_path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return float(json.load(f)["threshold"])
