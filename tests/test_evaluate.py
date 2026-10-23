import json

import numpy as np
import pytest

from lungai.config import load_config
from lungai.evaluate import compute_metrics, evaluate
from lungai.model import build_model


def test_compute_metrics_known_values():
    y_true = [0, 0, 1, 1, 1, 1]
    y_prob = [0.1, 0.7, 0.9, 0.8, 0.4, 0.6]
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert m["confusion_matrix"] == [[1, 1], [1, 3]]
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["specificity"] == pytest.approx(1 / 2)
    assert 0 <= m["roc_auc"] <= 1


def test_compute_metrics_single_class_has_no_auc():
    m = compute_metrics([1, 1, 1], [0.9, 0.2, 0.7])
    assert m["roc_auc"] is None


def test_evaluate_writes_reports(fake_dataset, tmp_path):
    config = load_config()
    config["data"].update(root=str(fake_dataset), image_size=[64, 64], batch_size=4)
    model = build_model("custom_cnn", image_size=(64, 64))

    metrics = evaluate(model, config, reports_dir=tmp_path / "reports")

    assert metrics["n_samples"] == 10
    for name in ("metrics.json", "confusion_matrix.png", "roc_curve.png"):
        assert (tmp_path / "reports" / name).exists()
    saved = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert np.sum(saved["confusion_matrix"]) == 10
