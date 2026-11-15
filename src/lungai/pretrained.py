"""Download the pretrained model published as a GitHub release.

Usage::

    python scripts/download_model.py

Files are checked against known SHA-256 hashes before they are saved, so a
partial or tampered download never lands in ``models/``.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

RELEASE_TAG = "v1.0.0"
RELEASE_URL = f"https://github.com/Lingesh-S/xray-diagnosis-cnn/releases/download/{RELEASE_TAG}"

# File name -> SHA-256 of the released asset.
RELEASE_FILES = {
    "mobilenetv2_ft.keras": "265614f5a91ab049ebdd912f4d0a829cd6fb4b7a26fdf0ddf793c4c542300776",
    "mobilenetv2_ft_threshold.json": "333eb57f028b950d810c6393c90e8de9e05d8da9a5fc52d4cb5dec42a9d264d7",
}


def release_url() -> str:
    """Base URL of the release assets (override with ``LUNGAI_RELEASE_URL``)."""
    return os.environ.get("LUNGAI_RELEASE_URL", RELEASE_URL).rstrip("/")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_downloaded(models_dir: str | Path) -> bool:
    """True when every release file is present in ``models_dir`` with the right hash."""
    models_dir = Path(models_dir)
    return all(
        (models_dir / name).exists() and sha256(models_dir / name) == expected
        for name, expected in RELEASE_FILES.items()
    )


def download_pretrained(models_dir: str | Path) -> list[Path]:
    """Download the released model files into ``models_dir`` and return their paths.

    Files that are already present with the right hash are not downloaded again.
    Raises ``ValueError`` if a downloaded file does not match its expected hash.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    base = release_url()
    saved = []
    for name, expected in RELEASE_FILES.items():
        target = models_dir / name
        if not (target.exists() and sha256(target) == expected):
            fd, tmp = tempfile.mkstemp(dir=models_dir, suffix=".part")
            os.close(fd)
            try:
                urllib.request.urlretrieve(f"{base}/{name}", tmp)
                actual = sha256(tmp)
                if actual != expected:
                    raise ValueError(f"Checksum mismatch for {name}: expected {expected}, got {actual}")
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        saved.append(target)
    return saved
