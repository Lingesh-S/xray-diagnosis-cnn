import numpy as np
import pytest

from lungai.gradcam import compute_gradcam, find_target_layer, overlay_heatmap
from lungai.model import MODEL_NAMES, build_model
from lungai.predict import predict_image, save_gradcam

from conftest import CLASSES


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_gradcam_heatmap(name):
    model = build_model(name, image_size=(96, 96), weights=None)
    image = np.random.default_rng(0).uniform(0, 255, size=(96, 96, 3)).astype("float32")

    heatmap, prob = compute_gradcam(model, image)

    assert heatmap.ndim == 2
    assert heatmap.min() >= 0 and heatmap.max() <= 1
    assert 0 <= prob <= 1
    assert prob == pytest.approx(float(model.predict(image[None], verbose=0)[0, 0]), abs=1e-4)


def test_target_layer_custom_cnn():
    model = build_model("custom_cnn", image_size=(64, 64))
    assert find_target_layer(model) == "block4_relu2"


def test_overlay_heatmap_shape():
    image = np.zeros((50, 60, 3), dtype=np.float32)
    out = overlay_heatmap(image, np.ones((7, 7)))
    assert out.shape == (50, 60, 3)
    assert out.dtype == np.uint8


def test_predict_image_and_gradcam_file(fake_dataset, tmp_path):
    model = build_model("custom_cnn", image_size=(64, 64))
    image_path = next((fake_dataset / "test" / "PNEUMONIA").iterdir())

    result = predict_image(model, image_path, CLASSES, (64, 64))
    assert result["label"] in CLASSES
    assert 0.5 <= result["confidence"] <= 1

    out = tmp_path / "cam.png"
    save_gradcam(model, image_path, (64, 64), out)
    assert out.exists() and out.stat().st_size > 0
