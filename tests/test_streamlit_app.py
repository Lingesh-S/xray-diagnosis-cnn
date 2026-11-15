from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from conftest import CLASSES
from lungai import pretrained
from lungai.model import build_model
from lungai.predict import interpret
from lungai.pretrained import sha256
from lungai.thresholds import save_threshold

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def test_interpret_threshold():
    assert interpret(0.7, CLASSES, 0.5) == {
        "label": "PNEUMONIA",
        "confidence": pytest.approx(0.7),
        "pneumonia_probability": 0.7,
    }
    result = interpret(0.7, CLASSES, 0.8)
    assert result["label"] == "NORMAL"
    assert result["confidence"] == pytest.approx(0.3)


def test_app_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNGAI_MODELS_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.warning[0].proto.title == "No trained model found"
    assert at.button[0].label == "Download pretrained model"
    assert not at.selectbox


def test_app_downloads_pretrained_model(tmp_path, monkeypatch):
    release = tmp_path / "release"
    release.mkdir()
    build_model("custom_cnn", image_size=(64, 64)).save(release / "mobilenetv2_ft.keras")
    save_threshold(release / "mobilenetv2_ft.keras", 0.81)
    monkeypatch.setattr(pretrained, "RELEASE_FILES", {p.name: sha256(p) for p in release.iterdir()})
    monkeypatch.setenv("LUNGAI_RELEASE_URL", release.as_uri())
    models = tmp_path / "models"
    monkeypatch.setenv("LUNGAI_MODELS_DIR", str(models))

    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button[0].click().run()

    assert not at.exception
    assert not at.error
    assert at.selectbox[0].value.name == "mobilenetv2_ft.keras"
    assert at.slider[0].value == pytest.approx(0.81)


def test_app_with_model(tmp_path, monkeypatch):
    build_model("custom_cnn", image_size=(64, 64)).save(tmp_path / "custom_cnn.keras")
    monkeypatch.setenv("LUNGAI_MODELS_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert not at.warning
    assert at.selectbox[0].value.name == "custom_cnn.keras"
    assert at.slider[0].value == 0.5


def test_app_uses_tuned_threshold(tmp_path, monkeypatch):
    model_path = tmp_path / "custom_cnn.keras"
    build_model("custom_cnn", image_size=(64, 64)).save(model_path)
    save_threshold(model_path, 0.72)
    monkeypatch.setenv("LUNGAI_MODELS_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.slider[0].value == pytest.approx(0.72)
