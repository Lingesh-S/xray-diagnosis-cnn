import pytest

from lungai.data import class_weights, list_images, make_dataset, make_splits

from conftest import CLASSES


def test_list_images_labels(fake_dataset):
    split = list_images(fake_dataset / "test", CLASSES)
    assert len(split) == 10
    assert split.labels.count(0) == 4
    assert split.labels.count(1) == 6


def test_make_splits_pools_and_stratifies(fake_dataset):
    splits = make_splits(fake_dataset, CLASSES, val_fraction=0.25, seed=1)
    # 12 normal + 32 pneumonia pooled from train/ and val/
    assert len(splits["train"]) + len(splits["val"]) == 44
    assert len(splits["val"]) == 11
    assert splits["val"].labels.count(0) == 3  # stratified ~27% normal
    assert set(splits["train"].paths).isdisjoint(splits["val"].paths)
    assert len(splits["test"]) == 10


def test_make_splits_is_deterministic(fake_dataset):
    a = make_splits(fake_dataset, CLASSES, seed=7)
    b = make_splits(fake_dataset, CLASSES, seed=7)
    assert a["val"].paths == b["val"].paths


def test_make_splits_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_splits(tmp_path, CLASSES)


def test_class_weights_upweight_minority():
    weights = class_weights([0] * 10 + [1] * 30)
    assert weights[0] > weights[1]
    assert weights[0] == pytest.approx(2.0)


@pytest.mark.parametrize("training", [False, True])
def test_make_dataset_shapes(fake_dataset, training):
    split = list_images(fake_dataset / "test", CLASSES)
    ds = make_dataset(split, (64, 64), batch_size=4, training=training)
    images, labels = next(iter(ds))
    assert images.shape == (4, 64, 64, 3)
    assert labels.shape == (4,)
    assert float(images.numpy().max()) <= 255.0 + 1e-3
