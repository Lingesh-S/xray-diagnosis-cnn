# Model Card — LungAI MobileNetV2 Pneumonia Classifier

## Model details
- **Model file:** `models/mobilenetv2_ft.keras` (not committed; reproduce with the README commands).
- **Architecture:** MobileNetV2 backbone pre-trained on ImageNet with a new head:
  global average pooling → dropout (0.3) → single sigmoid unit giving P(PNEUMONIA).
- **Input:** RGB chest X-ray resized to 224 × 224, raw pixel values in [0, 255]
  (rescaling to [-1, 1] happens inside the model).
- **Output:** probability of pneumonia. Decision threshold **0.81**, chosen on the validation split
  by maximising Youden's J (sensitivity + specificity − 1) and stored in
  `models/mobilenetv2_ft_threshold.json`.
- **Training:** two stages, both with binary cross-entropy, balanced class weights, light
  augmentation (rotation, zoom, translation, contrast) and early stopping on validation loss.
  1. Frozen backbone, head only: Adam lr 1e-3 (halved on plateau), 20 epochs.
  2. Fine-tuning: last 40 backbone layers unfrozen (BatchNorm kept frozen), Adam lr 1e-5, 10 epochs,
     best epoch by validation loss.
- **Framework:** TensorFlow / Keras. Config: [`configs/config.yaml`](../configs/config.yaml).

## Intended use
- Educational and portfolio demonstration of CNN-based medical image classification and
  Grad-CAM explainability.
- **Not intended** for clinical diagnosis, triage or any decision about a real patient.

## Data
- [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia),
  from Kermany et al., 2018 (Guangzhou Women and Children's Medical Center).
- 5,856 anterior-posterior chest X-rays of **pediatric patients aged 1–5**.
- Splits used:
  - Train + val folders pooled (5,232 images) and re-split 90/10, stratified, seed 42.
  - The official test folder (624 images: 234 normal, 390 pneumonia) is used only for final evaluation.

## Evaluation
Official test set, 624 images (234 normal, 390 pneumonia), threshold 0.81:

| Metric | Value |
|---|---|
| Accuracy | 89.1% |
| Sensitivity / pneumonia recall | 98.7% (385 / 390) |
| Specificity | 73.1% (171 / 234) |
| Precision | 85.9% |
| F1 | 0.919 |
| ROC-AUC | 0.974 |

Confusion matrix (rows = actual, columns = predicted):

|  | Normal | Pneumonia |
|---|---|---|
| **Normal** | 171 | 63 |
| **Pneumonia** | 5 | 385 |

Validation split (524 images): about 97% accuracy and 0.994 ROC-AUC. The gap to the test set is
a known property of this dataset, whose test images differ in distribution from the training
images. The project goal of 90% test accuracy was narrowly missed; the threshold was **not** tuned
on the test set to close it.

Comparison with earlier stages is in the [README](../README.md#results).

## Limitations
- **Population shift:** trained only on young children from a single hospital. Performance on
  adults, other hospitals, scanners or image formats is unknown and likely worse.
- **Binary only:** distinguishes normal vs pneumonia; it cannot detect other conditions and will
  still output one of the two labels for any image, including images that are not chest X-rays.
- **Bacterial vs viral** pneumonia are merged into one class.
- **Label noise:** the public dataset labels are known to contain some errors.
- **False alarms on normal images:** about 1 in 4 normal test X-rays is flagged as pneumonia.
- **Grad-CAM is approximate:** heatmaps show which regions influenced the score, not a
  verified localisation of disease. In some test images the heatmap also highlights the "R"
  side marker and image edges, a sign the model partly relies on such shortcuts.
- **Threshold choice** trades sensitivity against specificity; the default 0.5 is not tuned for
  any clinical setting.

## Ethical considerations
Any real-world use would need prospective validation on the target population, regulatory
approval and a clinician in the loop. The Streamlit app shows a "research use only" notice for
this reason.
