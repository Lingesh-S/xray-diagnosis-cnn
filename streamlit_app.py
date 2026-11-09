"""LungAI web demo: upload a chest X-ray, get a prediction and a Grad-CAM heatmap.

Run from the repository root::

    streamlit run streamlit_app.py

Trained models are read from ``models/*.keras`` (override with the
``LUNGAI_MODELS_DIR`` environment variable).
"""

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lungai.config import load_config  # noqa: E402
from lungai.thresholds import load_threshold  # noqa: E402

CONFIG = load_config()
CLASSES = CONFIG["data"]["classes"]
MODELS_DIR = Path(os.environ.get("LUNGAI_MODELS_DIR", ROOT / CONFIG["train"]["output_dir"]))


@st.cache_resource(max_entries=2, show_spinner="Loading model...")
def load_model(path: str, mtime: float):
    """Load a Keras model; ``mtime`` makes a retrained file reload."""
    import tensorflow as tf

    return tf.keras.models.load_model(path)


@st.cache_data(max_entries=16, show_spinner="Analyzing X-ray...")
def analyze(model_path: str, mtime: float, raw: bytes) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the resized image, its Grad-CAM heatmap and P(PNEUMONIA)."""
    from lungai.data import decode_image
    from lungai.gradcam import compute_gradcam

    model = load_model(model_path, mtime)
    image_size = tuple(model.input_shape[1:3])
    image = decode_image(raw, image_size).numpy()
    heatmap, prob = compute_gradcam(model, image)
    return image, heatmap, prob


st.set_page_config(page_title="LungAI", page_icon=":material/pulmonology:", layout="wide")

st.title("LungAI chest X-ray screening")
st.caption(
    "Upload a chest X-ray to classify it as normal or pneumonia and see which regions drove the prediction."
)
st.info(
    "This is an educational demo, not a medical device. Do not use it for diagnosis.",
    title="Research use only",
    icon=":material/info:",
)

# Newest model first, so a freshly trained or fine-tuned model is the default.
model_paths = sorted(MODELS_DIR.glob("*.keras"), key=lambda p: p.stat().st_mtime, reverse=True)
if not model_paths:
    st.warning(
        f"Train one first with `python -m lungai.train`; models are loaded from `{MODELS_DIR}`.",
        title="No trained model found",
        icon=":material/model_training:",
    )
    st.stop()

with st.sidebar:
    st.header("Settings")
    model_path = st.selectbox("Model", model_paths, format_func=lambda p: p.stem)
    threshold = st.slider(
        "Pneumonia threshold",
        min_value=0.05,
        max_value=0.95,
        value=round(load_threshold(model_path), 2),
        step=0.01,
        key=f"threshold_{model_path.stem}",
        help="Images with a pneumonia probability at or above this value are labelled pneumonia. "
        "Defaults to the threshold tuned on the validation set, if one was saved.",
    )
    show_heatmap = st.toggle("Show Grad-CAM heatmap", value=True)
    opacity = st.slider("Heatmap opacity", 0.1, 0.9, 0.4, 0.05, disabled=not show_heatmap)

uploaded = st.file_uploader("Chest X-ray image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.stop()

try:
    image, heatmap, prob = analyze(str(model_path), model_path.stat().st_mtime, uploaded.getvalue())
except Exception:  # corrupt or unsupported file
    st.error("Could not read this file as an image. Please upload a JPEG or PNG.", icon=":material/error:")
    st.stop()

from lungai.gradcam import overlay_heatmap  # noqa: E402
from lungai.predict import interpret  # noqa: E402

result = interpret(prob, CLASSES, threshold)

with st.container(horizontal=True):
    st.metric("Prediction", result["label"].capitalize(), border=True)
    st.metric("Confidence", f"{result['confidence']:.1%}", border=True)
    st.metric("Pneumonia probability", f"{prob:.1%}", border=True)

original_col, heatmap_col = st.columns(2)
with original_col:
    st.image(image.astype(np.uint8), caption="Uploaded X-ray", alt="Uploaded chest X-ray", width="stretch")
with heatmap_col:
    if show_heatmap:
        st.image(
            overlay_heatmap(image, heatmap, alpha=opacity),
            caption="Grad-CAM: warmer colours had more influence on the prediction",
            alt="Grad-CAM heatmap over the chest X-ray",
            width="stretch",
        )
