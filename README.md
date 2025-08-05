# LungAI: Chest X-Ray Classification with Deep Learning 

This project uses Convolutional Neural Networks (CNNs) to classify chest X-ray images into categories such as Normal, Pneumonia, and other lung abnormalities. It leverages deep learning techniques and optionally supports Grad-CAM visualization and interactive prediction via Streamlit.

##  Features

-  Chest X-ray image classification (Normal, Pneumonia)
-  CNN-based architecture (custom or transfer learning)
-  Training & evaluation metrics (accuracy, confusion matrix, etc.)
-  Grad-CAM visualization for model explainability (optional)
-  Optional Streamlit app for real-time X-ray predictions


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


---

## 🌐 Streamlit Web App *(Optional)*

Run the web app locally to upload your own X-ray images and get instant diagnosis:

```bash
streamlit run streamlit_app.py
```

## 📄 License
This project is open source and available under the MIT License.

# 👨‍💻 Author
**Lingesh S**

[GitHub](https://github.com/Lingesh-S) • [LinkedIn](https://linkedin.com/in/lingesh-s29)
