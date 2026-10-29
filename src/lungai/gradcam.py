"""Grad-CAM heatmaps showing which regions drove a prediction.

Works with every model from ``lungai.model``: they are all a linear chain of
layers, so the model is split at the last 4D feature map (the backbone output
for transfer models, the last conv activation for the custom CNN) and the
gradient of the predicted class score is taken with respect to that map.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from matplotlib import colormaps


def find_target_layer(model: tf.keras.Model) -> str:
    """Name of the last layer that outputs a spatial (4D) feature map."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.MaxPooling2D):
            continue
        shape = getattr(layer, "output", None)
        if shape is not None and len(shape.shape) == 4 and not isinstance(layer, tf.keras.layers.InputLayer):
            return layer.name
    raise ValueError("No convolutional feature map found in the model.")


def compute_gradcam(
    model: tf.keras.Model, image: np.ndarray, layer_name: str | None = None
) -> tuple[np.ndarray, float]:
    """Return a heatmap in [0, 1] (feature-map resolution) and P(PNEUMONIA).

    ``image`` is a single RGB image in [0, 255] with shape (H, W, 3).
    """
    layer_name = layer_name or find_target_layer(model)
    names = [layer.name for layer in model.layers]
    split = names.index(layer_name)

    x = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        for layer in model.layers[1 : split + 1]:
            x = layer(x, training=False)
        features = x
        tape.watch(features)
        for layer in model.layers[split + 1 :]:
            x = layer(x, training=False)
        prob = x[0, 0]
        # Explain whichever class the model predicted.
        score = tf.where(prob >= 0.5, prob, 1.0 - prob)

    grads = tape.gradient(score, features)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(features[0] * weights, axis=-1)).numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, float(prob)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a [0, 1] heatmap (any size) over an RGB image; returns uint8 RGB."""
    h, w = image.shape[:2]
    resized = tf.image.resize(heatmap[..., None], (h, w)).numpy()[..., 0]
    colored = colormaps["jet"](np.clip(resized, 0, 1))[..., :3] * 255.0
    blended = (1 - alpha) * image.astype(np.float32) + alpha * colored
    return np.clip(blended, 0, 255).astype(np.uint8)
