import numpy as np
import pytest

from lungai.model import MODEL_NAMES, build_model


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_build_model_output(name):
    model = build_model(name, image_size=(96, 96), weights=None)
    x = np.random.uniform(0, 255, size=(2, 96, 96, 3)).astype("float32")
    y = model.predict(x, verbose=0)
    assert y.shape == (2, 1)
    assert np.all((y >= 0) & (y <= 1))


def test_transfer_backbone_frozen_by_default():
    model = build_model("mobilenetv2", image_size=(96, 96), weights=None)
    backbone = next(l for l in model.layers if l.name.startswith("mobilenetv2"))
    assert not backbone.trainable


def test_fine_tune_unfreezes_last_layers():
    model = build_model("mobilenetv2", image_size=(96, 96), weights=None, fine_tune_layers=10)
    backbone = next(l for l in model.layers if l.name.startswith("mobilenetv2"))
    trainable = [l for l in backbone.layers if l.trainable and l.weights]
    assert trainable
    assert all(l in backbone.layers[-10:] for l in trainable)


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model("vgg_nope", weights=None)
