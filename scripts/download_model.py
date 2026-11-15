"""Download the pretrained LungAI model from the GitHub release into ``models/``.

Usage::

    python scripts/download_model.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lungai.pretrained import RELEASE_TAG, download_pretrained  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default=REPO_ROOT / "models", type=Path)
    args = parser.parse_args()

    print(f"Downloading pretrained model ({RELEASE_TAG}) into {args.dest} ...")
    for path in download_pretrained(args.dest):
        print(f"  {path.name}")
    print("Done. Run `streamlit run streamlit_app.py` to try it.")


if __name__ == "__main__":
    main()
