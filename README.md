# 🧠 Brain Tumor Detection Using Deep Learning and Explainable AI

## 📌 Project Overview

Brain tumors are among the most critical neurological disorders that require early and accurate diagnosis. This project presents an AI-powered system that automatically detects brain tumors from MRI images using a Convolutional Neural Network (CNN) with MobileNetV2 Transfer Learning. The system also integrates Explainable AI (Grad-CAM) to visualize the regions that influenced the model's prediction.

The application is deployed as a web-based system using Flask, allowing users to upload MRI images and receive real-time predictions along with visual explanations.

---

## 🎯 Objectives

* Detect brain tumors from MRI images automatically.
* Improve diagnostic accuracy using deep learning.
* Provide real-time prediction through a web application.
* Integrate Explainable AI (Grad-CAM) for transparency.
* Assist medical professionals in decision-making.

---

## 🏗️ System Architecture

MRI Image Input

↓

Image Preprocessing

↓

MobileNetV2 Feature Extraction

↓

Custom CNN Layers

↓

Tumor / No Tumor Classification

↓

Grad-CAM Heatmap Generation

↓

Web Application Output

---

## 📂 Dataset Description

The dataset consists of MRI brain scan images categorized into two classes:

* Tumor
* No Tumor

Dataset Structure:

dataset/

├── Training/

│ ├── tumor/

│ └── no_tumor/

│

└── Testing/

├── tumor/

└── no_tumor/

Images are resized to 224 × 224 pixels and normalized before training.

---

## 🧠 Model Architecture

### Pre-trained Model

* MobileNetV2 (Transfer Learning)

### Custom Layers

1. GlobalAveragePooling2D
2. Dense Layer (128 neurons, ReLU)
3. Dropout Layer (0.5)
4. Output Layer (Sigmoid)

### Compilation Parameters

Optimizer:

* Adam

Loss Function:

* Binary Crossentropy

Metric:

* Accuracy

---

## ⚙️ Technologies Used

### Programming Language

* Python

### Frameworks & Libraries

* TensorFlow / Keras
* NumPy
* Pandas
* OpenCV
* Matplotlib
* Flask
* Pillow
* Scikit-learn

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Flask REST API

---

## 🚀 Installation

### Create Virtual Environment

Windows:

python -m venv tf_env

tf_env\Scripts\activate

### Install Dependencies

pip install tensorflow flask flask-cors numpy pillow matplotlib opencv-python scikit-learn

---

## ▶️ Running the Project

### Step 1: Train the Model

python train_model.py

This generates:

brain_tumor_model.h5

Move the model into:

backend/model/

### Step 2: Start Backend

cd backend

python app.py

Server runs at:

http://127.0.0.1:5000

### Step 3: Launch Frontend

Open:

frontend/index.html

or use VS Code Live Server.

---

## 🔍 Features

* MRI Image Upload
* Brain Tumor Detection
* Confidence Score Display
* Grad-CAM Heatmap Visualization
* Real-Time Prediction
* User-Friendly Interface

---

## 📊 Performance Metrics

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 94.66% |
| Loss      | 0.1707 |
| Precision | 95%    |
| Recall    | 94%    |
| F1-Score  | 94.5%  |

---

## 🧪 Explainable AI (Grad-CAM)

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights important regions in MRI images that contribute to the model's prediction.

Benefits:

* Improves transparency
* Enhances trust in AI predictions
* Assists medical interpretation

---

## 📈 Results Achieved

* Successfully classified MRI images into Tumor and No Tumor categories.
* Achieved 94.66% classification accuracy.
* Implemented Explainable AI using Grad-CAM.
* Developed a real-time web application for prediction.
* Reduced dependency on manual image analysis.

---

## 🔮 Future Enhancements

* Multi-class brain tumor classification
* Cloud deployment
* Patient report generation (PDF)
* Integration with hospital management systems
* Advanced explainable AI techniques

---

## 📚 References

1. The Cancer Imaging Archive (TCIA)
2. TensorFlow Documentation
3. Scikit-learn Documentation
4. Deep Learning by Goodfellow, Bengio, and Courville
5. Grad-CAM Research Paper (Selvaraju et al., 2017)

---

## 👨‍💻 Author

Brain Tumor Detection Using Deep Learning and Explainable AI

Academic Mini/Major Project
