from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from conftest import CLASSES
from lungai.model import build_model
from lungai.predict import interpret

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
    assert not at.selectbox


def test_app_with_model(tmp_path, monkeypatch):
    build_model("custom_cnn", image_size=(64, 64)).save(tmp_path / "custom_cnn.keras")
    monkeypatch.setenv("LUNGAI_MODELS_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert not at.warning
    assert at.selectbox[0].value.name == "custom_cnn.keras"
    assert at.slider[0].value == 0.5
