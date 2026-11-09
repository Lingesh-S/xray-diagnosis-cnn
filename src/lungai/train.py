"""Training entry point.

Usage::

    python -m lungai.train                       # uses configs/config.yaml
    python -m lungai.train --model custom_cnn --epochs 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from lungai.config import load_config
from lungai.data import class_weights, make_dataset, make_splits
from lungai.model import MODEL_NAMES, build_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def train(
    config: dict,
    weights: str | None = "imagenet",
    init_from: str | Path | None = None,
    run_name: str | None = None,
) -> tuple[tf.keras.Model, dict]:
    """Train a model as described by ``config``; returns the model and its history.

    ``init_from`` starts from a saved model's weights, which is how the second,
    fine-tuning stage continues from the frozen-backbone run. Outputs are named
    after ``run_name`` (defaults to the model name).
    """
    seed = config["seed"]
    tf.keras.utils.set_random_seed(seed)

    data_cfg, model_cfg, train_cfg = config["data"], config["model"], config["train"]
    image_size = tuple(data_cfg["image_size"])

    splits = make_splits(data_cfg["root"], data_cfg["classes"], data_cfg["val_fraction"], seed)
    train_ds = make_dataset(splits["train"], image_size, data_cfg["batch_size"], training=True, seed=seed)
    val_ds = make_dataset(splits["val"], image_size, data_cfg["batch_size"])
    print(f"Images: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    model = build_model(
        model_cfg["name"],
        image_size=image_size,
        dropout=model_cfg["dropout"],
        fine_tune_layers=model_cfg["fine_tune_layers"],
        weights=weights,
    )
    if init_from:
        model.load_weights(init_from)
    compile_model(model, train_cfg["learning_rate"])

    run_name = run_name or model_cfg["name"]
    out_dir = Path(train_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{run_name}.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=train_cfg["early_stopping_patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        tf.keras.callbacks.CSVLogger(out_dir / f"{run_name}_history.csv"),
    ]

    cw = class_weights(splits["train"].labels) if train_cfg["use_class_weights"] else None
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=train_cfg["epochs"],
        class_weight=cw,
        callbacks=callbacks,
    )

    with open(out_dir / f"{run_name}_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Best model saved to {model_path}")
    return model, history.history


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a chest X-ray pneumonia classifier.")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument("--model", choices=MODEL_NAMES, help="Override model.name.")
    parser.add_argument("--epochs", type=int, help="Override train.epochs.")
    parser.add_argument("--data-root", help="Override data.root.")
    parser.add_argument("--learning-rate", type=float, help="Override train.learning_rate.")
    parser.add_argument(
        "--fine-tune-layers", type=int, help="Override model.fine_tune_layers (unfreeze last N layers)."
    )
    parser.add_argument("--init-from", help="Start from this saved model's weights (fine-tuning).")
    parser.add_argument("--run-name", help="Output file name stem; defaults to the model name.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.model:
        config["model"]["name"] = args.model
    if args.epochs:
        config["train"]["epochs"] = args.epochs
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.learning_rate:
        config["train"]["learning_rate"] = args.learning_rate
    if args.fine_tune_layers is not None:
        config["model"]["fine_tune_layers"] = args.fine_tune_layers
    train(config, init_from=args.init_from, run_name=args.run_name)


if __name__ == "__main__":
    main()
