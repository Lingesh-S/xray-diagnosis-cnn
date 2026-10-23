"""Dataset discovery, splitting and ``tf.data`` pipelines.

Expected layout::

    <root>/train/<CLASS>/*.jpeg
    <root>/val/<CLASS>/*.jpeg     (optional, merged into train then re-split)
    <root>/test/<CLASS>/*.jpeg
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class Split:
    paths: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.paths)


def list_images(split_dir: str | Path, classes: list[str]) -> Split:
    """Collect image paths and integer labels (index into ``classes``)."""
    split_dir = Path(split_dir)
    paths, labels = [], []
    for label, cls in enumerate(classes):
        class_dir = split_dir / cls
        if not class_dir.is_dir():
            continue
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(p))
                labels.append(label)
    return Split(paths, labels)


def make_splits(
    root: str | Path, classes: list[str], val_fraction: float = 0.1, seed: int = 42
) -> dict[str, Split]:
    """Build train/val/test splits.

    The official Kaggle validation folder only has 16 images, so train and val
    are pooled and a stratified, seeded validation split is carved out instead.
    """
    root = Path(root)
    train = list_images(root / "train", classes)
    val = list_images(root / "val", classes)
    test = list_images(root / "test", classes)

    pooled_paths = train.paths + val.paths
    pooled_labels = train.labels + val.labels
    if not pooled_paths:
        raise FileNotFoundError(
            f"No training images found under {root}. Run scripts/download_data.py first."
        )

    tr_p, va_p, tr_y, va_y = train_test_split(
        pooled_paths,
        pooled_labels,
        test_size=val_fraction,
        stratify=pooled_labels,
        random_state=seed,
    )
    return {"train": Split(tr_p, tr_y), "val": Split(va_p, va_y), "test": test}


def class_weights(labels: list[int]) -> dict[int, float]:
    """Balanced class weights to counter the ~3:1 pneumonia/normal imbalance."""
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=np.asarray(labels))
    return {int(c): float(w) for c, w in zip(classes, weights)}


def load_image(path: tf.Tensor, image_size: tuple[int, int]) -> tf.Tensor:
    """Read an image file as a float32 RGB tensor in [0, 255]."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size)
    return tf.cast(img, tf.float32)


def augmentation() -> tf.keras.Sequential:
    """Light, anatomy-preserving augmentation (no vertical flips)."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def make_dataset(
    split: Split,
    image_size: tuple[int, int],
    batch_size: int,
    training: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Create a batched ``tf.data.Dataset`` yielding ``(image, label)`` pairs.

    Images stay in [0, 255]; each model applies its own preprocessing.
    """
    image_size = tuple(image_size)
    ds = tf.data.Dataset.from_tensor_slices(
        (split.paths, np.asarray(split.labels, dtype=np.float32))
    )
    if training:
        ds = ds.shuffle(len(split), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (load_image(p, image_size), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if training:
        aug = augmentation()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
