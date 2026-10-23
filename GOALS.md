# LungAI — Project Goals & Roadmap

**Repository:** `xray-diagnosis-cnn`
**Owner:** Lingesh S
**Started:** 2026-10-21

---

## 1. Problem Statement

Pneumonia is one of the leading causes of hospitalisation worldwide, and chest
X-rays are the most common first-line imaging exam used to detect it. Reading
X-rays takes trained radiologists, who are in short supply in many regions.

**LungAI** is an end-to-end deep learning project that classifies chest X-ray
images as **NORMAL** or **PNEUMONIA** with a Convolutional Neural Network, shows
*why* the model made its decision with Grad-CAM heatmaps, and exposes the model
through a simple Streamlit web app.

> ⚠️ This is an educational / portfolio project. It is **not** a medical device
> and must not be used for real clinical diagnosis.

---

## 2. Goals

| # | Goal | Success metric |
|---|------|----------------|
| G1 | Reproducible data pipeline for the Kaggle *Chest X-Ray Images (Pneumonia)* dataset | One command downloads/organises data; train/val/test splits are deterministic |
| G2 | Baseline custom CNN | Test accuracy ≥ 85% |
| G3 | Transfer-learning model (MobileNetV2 / EfficientNetB0) | Test accuracy ≥ 90%, pneumonia recall ≥ 95% |
| G4 | Proper evaluation | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix saved to `reports/` |
| G5 | Explainability | Grad-CAM heatmap overlay for any input image |
| G6 | Interactive demo | `streamlit run streamlit_app.py` → upload image → prediction + confidence + heatmap |
| G7 | Engineering quality | Modular `src/` package, config file, unit tests, GitHub Actions CI |
| G8 | Documentation | README with setup, training, results and screenshots |

### Non-goals
- Multi-disease classification beyond Normal/Pneumonia (possible future work).
- Clinical validation or regulatory compliance.
- Shipping trained weights inside git (weights are produced locally / released separately).

---

## 3. Dataset

- **Source:** [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
  (Kermany et al., 2018)
- ~5,856 pediatric chest X-ray JPEGs, 2 classes: `NORMAL`, `PNEUMONIA`.
- Known issues handled by the pipeline:
  - **Class imbalance** (~3:1 pneumonia:normal) → class weights.
  - **Tiny official validation split** (16 images) → re-split train into train/val (90/10, stratified, fixed seed).

Expected layout after download (`data/` is git-ignored):

```
data/chest_xray/
├── train/{NORMAL,PNEUMONIA}/
├── val/{NORMAL,PNEUMONIA}/
└── test/{NORMAL,PNEUMONIA}/
```

---

## 4. Architecture

```
xray-diagnosis-cnn/
├── configs/config.yaml        # all hyper-parameters & paths
├── src/lungai/
│   ├── config.py              # config loader
│   ├── data.py                # dataset discovery, splitting, tf.data pipelines, augmentation
│   ├── model.py               # custom CNN + transfer-learning builders
│   ├── train.py               # training entry point (callbacks, class weights)
│   ├── evaluate.py            # metrics + plots            (Phase 2)
│   ├── gradcam.py             # Grad-CAM explainability    (Phase 2)
│   └── predict.py             # single-image inference     (Phase 2)
├── scripts/download_data.py   # Kaggle download helper
├── streamlit_app.py           # web demo                   (Phase 3)
├── tests/                     # pytest unit tests
└── .github/workflows/ci.yml   # lint + tests               (Phase 3)
```

**Stack:** Python 3.10+, TensorFlow/Keras, NumPy, OpenCV, Matplotlib,
scikit-learn, Streamlit, pytest.

---

## 5. Roadmap

Work happens on feature branches, merged into `main` via pull requests.
Pace: at most 3–4 commits per working day; a progress report every 2 days in
[`docs/progress/`](docs/progress/).

| Phase | Branch | Scope | Status |
|-------|--------|-------|--------|
| 1 — Foundation | `feature/project-foundation` | Goals doc, project layout, config, data pipeline, CNN models, training script, tests | 🚧 On branch, awaiting PR |
| 2 — Evaluation & Explainability | `feature/evaluation-gradcam` | `evaluate.py`, metrics/plots, `gradcam.py`, `predict.py` CLI, tests | ⏳ Planned |
| 3 — Web App & CI | `feature/streamlit-app` | `streamlit_app.py`, GitHub Actions CI (lint + tests) | ⏳ Planned |
| 4 — Polish | `feature/docs-results` | README rewrite, results table, screenshots, model card | ⏳ Planned |

### Definition of done (whole project)
- [ ] `python scripts/download_data.py` prepares the dataset
- [ ] `python -m lungai.train --model mobilenetv2` trains and saves a model
- [ ] `python -m lungai.evaluate` writes metrics + confusion matrix to `reports/`
- [ ] `python -m lungai.predict path/to/xray.jpeg` prints label + confidence
- [ ] `streamlit run streamlit_app.py` serves the demo with Grad-CAM
- [ ] CI is green on every PR
