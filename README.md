# LungAI: Chest X-Ray Classification with Deep Learning 

This project uses Convolutional Neural Networks (CNNs) to classify chest X-ray images into categories such as Normal, Pneumonia, and other lung abnormalities. It leverages deep learning techniques and optionally supports Grad-CAM visualization and interactive prediction via Streamlit.

## Features
- Multi-class classification: Normal, Pneumonia, etc.
- Built using custom CNN and/or transfer learning (e.g., ResNet, MobileNet)
- Model explainability with Grad-CAM
- Optional Streamlit app for real-time prediction from uploaded X-rays


---

## 🏗️ Tech Stack

- Python
- OpenCV
- TensorFlow / Keras
- Matplotlib, NumPy
- Streamlit *(optional UI)*
- Grad-CAM *(for explainability)*

---

## 📈 Model Training & Evaluation

- Custom CNN architecture or pretrained model (ResNet/MobileNet)
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy, Precision, Recall, F1 Score

---

## 🖼️ Sample Predictions

| Image | Prediction |
|-------|------------|
| Chest X-ray 1 | Pneumonia |
| Chest X-ray 2 | Normal |

---

## 🔍 Grad-CAM Visualizations *(Optional)*

Grad-CAM is used to highlight regions in the image that influenced the model's prediction.

![gradcam-sample](./assets/gradcam-example.png)

---

## 🌐 Streamlit Web App *(Optional)*

Run the web app locally to upload your own X-ray images and get instant diagnosis:

```bash
streamlit run streamlit_app.py
```

#📄 License
This project is open source and available under the MIT License.

#🙋‍♂️ Author
Lingesh S

