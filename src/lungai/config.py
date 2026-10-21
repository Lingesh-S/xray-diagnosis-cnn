"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config file (defaults to ``configs/config.yaml``)."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
