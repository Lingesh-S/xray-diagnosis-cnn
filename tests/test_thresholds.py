import numpy as np
import pytest

from lungai.evaluate import select_threshold
from lungai.thresholds import load_threshold, save_threshold, threshold_path


def test_select_threshold_separates_classes():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.2, 0.6, 0.65, 0.8, 0.9, 0.95])
    threshold = select_threshold(y_true, y_prob)
    assert 0.65 < threshold <= 0.8
    assert np.all((y_prob >= threshold) == y_true.astype(bool))


def test_threshold_round_trip(tmp_path):
    model_path = tmp_path / "mobilenetv2.keras"
    assert load_threshold(model_path) == 0.5
    assert load_threshold(model_path, default=0.3) == 0.3

    save_threshold(model_path, 0.72)

    assert threshold_path(model_path).name == "mobilenetv2_threshold.json"
    assert load_threshold(model_path) == pytest.approx(0.72)
