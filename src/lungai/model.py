"""Model builders: a from-scratch CNN baseline and transfer-learning models.

Every model takes raw RGB images in [0, 255] and outputs P(PNEUMONIA) through a
single sigmoid unit, so inference code never has to know which backbone is used.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

MODEL_NAMES = ("custom_cnn", "mobilenetv2", "efficientnetb0")


def _conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return layers.MaxPooling2D(name=f"{name}_pool")(x)


def build_custom_cnn(input_shape: tuple[int, int, int], dropout: float = 0.3) -> tf.keras.Model:
    """Baseline CNN: four conv blocks + global pooling + dense head."""
    inputs = layers.Input(shape=input_shape, name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)
    for i, filters in enumerate((32, 64, 128, 256), start=1):
        x = _conv_block(x, filters, name=f"block{i}")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    x = layers.Dense(128, activation="relu", name="fc")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pneumonia_prob")(x)
    return tf.keras.Model(inputs, outputs, name="custom_cnn")


def build_transfer_model(
    backbone_name: str,
    input_shape: tuple[int, int, int],
    dropout: float = 0.3,
    fine_tune_layers: int = 0,
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    """ImageNet backbone + new classification head.

    The backbone is frozen except for its last ``fine_tune_layers`` layers
    (BatchNorm layers always stay frozen for stable fine-tuning).
    """
    inputs = layers.Input(shape=input_shape, name="image")
    if backbone_name == "mobilenetv2":
        # Equivalent to mobilenet_v2.preprocess_input: [0, 255] -> [-1, 1]
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(inputs)
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False, weights=weights, input_shape=input_shape
        )
    elif backbone_name == "efficientnetb0":
        # EfficientNet rescales internally and expects [0, 255] input.
        x = inputs
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=weights, input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown model '{backbone_name}'. Choose from {MODEL_NAMES}.")

    backbone.trainable = fine_tune_layers > 0
    if fine_tune_layers > 0:
        for layer in backbone.layers[:-fine_tune_layers]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pneumonia_prob")(x)
    return tf.keras.Model(inputs, outputs, name=backbone_name)


def build_model(
    name: str,
    image_size: tuple[int, int] = (224, 224),
    dropout: float = 0.3,
    fine_tune_layers: int = 0,
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    """Build any supported model by name."""
    input_shape = (*image_size, 3)
    if name == "custom_cnn":
        return build_custom_cnn(input_shape, dropout)
    return build_transfer_model(name, input_shape, dropout, fine_tune_layers, weights)
