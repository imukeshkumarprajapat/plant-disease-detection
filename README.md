# 🌱 Plant Disease Detection using Deep Learning

A Streamlit-based web application to detect **Potato leaf diseases** (Healthy, Early Blight, Late Blight) using a trained CNN model.  
This project demonstrates end-to-end workflow: **data preprocessing → model training → evaluation → deployment**.

---

## 🚀 Features
- Upload leaf images and get instant disease predictions
- Confidence score for each prediction
- Interactive dashboard with:
  - Confusion Matrix visualization
  - Validation vs Test dataset performance comparison
- Easy deployment with Streamlit
- Lightweight `.h5` model format for portability

---

## 📊 Dataset
- Source: PlantVillage dataset (Potato leaves)
- Classes:
  - `Healthy`
  - `Potato___Early_blight`
  - `Potato___Late_blight`
- Total samples: ~2,152 images (split into train, validation, test)

---

## 🧠 Model Architecture
- Convolutional Neural Network (CNN) built with TensorFlow/Keras
- Layers:
  - Conv2D + MaxPooling (multiple blocks)
  - Flatten + Dense layers
  - Softmax output for 3 classes
- Data Augmentation: Random Flip, Random Rotation
- Optimizer: Adam
- Loss: SparseCategoricalCrossentropy

---

## 📈 Performance Summary

| Dataset     | Total Samples | Correct Predictions | Wrong Predictions | Accuracy |
|-------------|---------------|----------------------|-------------------|----------|
| Validation  | 192           | 186                  | 6                 | 96.88%   |
| Test        | 232           | 227                  | 5                 | 97.84%   |

- Confusion Matrix and F1-score charts included in Streamlit app  
- Validation vs Test performance comparison with annotated bar charts


## 🖥️ Streamlit App

### 🔎 How to Run
1. Save trained model as `.h5`:
   ```python
   model.save("plant_disease_model.h5")


2. Install dependencies:

pip install -r requirements.txt


3. Run Streamlit:
streamlit run app.py

📂 Project Structure

plant_disease_app/
│── app.py
│── plant_disease_model.h5
│── requirements.txt
│── README.md


📦 Requirements

streamlit
tensorflow
numpy
pillow
scikit-learn
matplotlib


