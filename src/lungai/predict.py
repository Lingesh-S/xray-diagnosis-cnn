"""Predict on a single chest X-ray image.

Usage::

    python -m lungai.predict path/to/xray.jpeg
    python -m lungai.predict path/to/xray.jpeg --gradcam heatmap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from lungai.config import load_config
from lungai.data import load_image
from lungai.gradcam import compute_gradcam, overlay_heatmap


def predict_image(
    model: tf.keras.Model,
    image_path: str | Path,
    classes: list[str],
    image_size: tuple[int, int],
    threshold: float = 0.5,
) -> dict:
    """Classify one image; returns label, confidence and P(PNEUMONIA)."""
    image = load_image(tf.constant(str(image_path)), tuple(image_size)).numpy()
    prob = float(model.predict(image[None, ...], verbose=0)[0, 0])
    is_positive = prob >= threshold
    return {
        "label": classes[1] if is_positive else classes[0],
        "confidence": prob if is_positive else 1.0 - prob,
        "pneumonia_probability": prob,
    }


def save_gradcam(
    model: tf.keras.Model, image_path: str | Path, image_size: tuple[int, int], out_path: str | Path
) -> None:
    image = load_image(tf.constant(str(image_path)), tuple(image_size)).numpy()
    heatmap, _ = compute_gradcam(model, image)
    overlay = overlay_heatmap(image, heatmap)
    tf.io.write_file(str(out_path), tf.io.encode_png(overlay))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict NORMAL / PNEUMONIA for one X-ray.")
    parser.add_argument("image", help="Path to a chest X-ray image.")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument("--model-path", help="Defaults to <train.output_dir>/<model.name>.keras.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gradcam", help="Optional output path for a Grad-CAM overlay PNG.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    model_path = args.model_path or Path(config["train"]["output_dir"]) / f"{config['model']['name']}.keras"
    model = tf.keras.models.load_model(model_path)
    image_size = tuple(config["data"]["image_size"])

    result = predict_image(model, args.image, config["data"]["classes"], image_size, args.threshold)
    print(f"{result['label']} ({result['confidence']:.1%} confidence)")

    if args.gradcam:
        save_gradcam(model, args.image, image_size, args.gradcam)
        print(f"Grad-CAM overlay saved to {args.gradcam}")


if __name__ == "__main__":
    main()
