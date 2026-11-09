"""Evaluate a trained model on the test split.

Usage::

    python -m lungai.evaluate                          # models/<model.name>.keras
    python -m lungai.evaluate --model-path models/custom_cnn.keras --threshold 0.5

Writes ``metrics.json``, ``confusion_matrix.png`` and ``roc_curve.png`` to the
reports directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from lungai.config import load_config  # noqa: E402
from lungai.data import Split, list_images, make_dataset, make_splits  # noqa: E402
from lungai.thresholds import save_threshold  # noqa: E402


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Classification metrics for binary labels (1 = PNEUMONIA)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "threshold": threshold,
        "n_samples": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in metrics.items()}


def plot_confusion_matrix(cm: list[list[int]], classes: list[str], path: Path) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, auc: float | None, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}" if auc is not None else None)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    if auc is not None:
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold that maximises Youden's J (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best = int(np.argmax(tpr - fpr))
    return float(np.clip(thresholds[best], 0.0, 1.0))


def predict_split(model: tf.keras.Model, split: Split, config: dict) -> tuple[np.ndarray, np.ndarray]:
    data_cfg = config["data"]
    ds = make_dataset(split, tuple(data_cfg["image_size"]), data_cfg["batch_size"])
    return np.asarray(split.labels), model.predict(ds, verbose=0).ravel()


def tune_threshold(model: tf.keras.Model, config: dict) -> float:
    """Pick the decision threshold on the validation split (never on the test set)."""
    data_cfg = config["data"]
    splits = make_splits(data_cfg["root"], data_cfg["classes"], data_cfg["val_fraction"], config["seed"])
    y_true, y_prob = predict_split(model, splits["val"], config)
    return select_threshold(y_true, y_prob)


def evaluate(
    model: tf.keras.Model, config: dict, reports_dir: str | Path = "reports", threshold: float = 0.5
) -> dict:
    """Run the model on the test split and write metrics and plots."""
    data_cfg = config["data"]
    test = list_images(Path(data_cfg["root"]) / "test", data_cfg["classes"])
    if not len(test):
        raise FileNotFoundError(f"No test images found under {data_cfg['root']}/test")

    y_true, y_prob = predict_split(model, test, config)
    metrics = compute_metrics(y_true, y_prob, threshold)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    plot_confusion_matrix(
        metrics["confusion_matrix"], data_cfg["classes"], reports_dir / "confusion_matrix.png"
    )
    if metrics["roc_auc"] is not None:
        plot_roc_curve(y_true, y_prob, metrics["roc_auc"], reports_dir / "roc_curve.png")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split.")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument("--model-path", help="Defaults to <train.output_dir>/<model.name>.keras.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Choose the threshold on the validation split and save it next to the model.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    model_path = Path(
        args.model_path or Path(config["train"]["output_dir"]) / f"{config['model']['name']}.keras"
    )
    model = tf.keras.models.load_model(model_path)

    threshold = args.threshold
    if args.tune_threshold:
        threshold = tune_threshold(model, config)
        save_threshold(model_path, threshold)
        print(f"   threshold: {threshold:.4f} (chosen on validation split)")

    metrics = evaluate(model, config, args.reports_dir, threshold)

    for key in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc"):
        value = metrics[key]
        print(f"{key:>12}: {value:.4f}" if value is not None else f"{key:>12}: n/a")
    print(f"Reports written to {args.reports_dir}/")


if __name__ == "__main__":
    main()
