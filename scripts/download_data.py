"""Download the Kaggle Chest X-Ray Images (Pneumonia) dataset into ``data/``.

Requires Kaggle credentials (``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME``
and ``KAGGLE_KEY`` environment variables).

Usage::

    python scripts/download_data.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DATASET = "paultimothymooney/chest-xray-pneumonia"
REPO_ROOT = Path(__file__).resolve().parents[1]


def find_dataset_root(base: Path) -> Path:
    """Locate the folder that contains train/ and test/ inside the download."""
    for candidate in [base, *sorted(p for p in base.rglob("*") if p.is_dir())]:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find train/ and test/ under {base}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default=REPO_ROOT / "data" / "chest_xray", type=Path)
    args = parser.parse_args()

    if (args.dest / "train").is_dir():
        print(f"Dataset already present at {args.dest}")
        return

    import kagglehub

    download_dir = Path(kagglehub.dataset_download(DATASET))
    src = find_dataset_root(download_dir)
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, args.dest, ignore=shutil.ignore_patterns("__MACOSX", ".DS_Store"))
    print(f"Dataset copied to {args.dest}")


if __name__ == "__main__":
    main()
