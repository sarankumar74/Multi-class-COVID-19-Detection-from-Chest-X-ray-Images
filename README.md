# 🩺 Multi-Class COVID-19 Detection from Chest X-ray Images
🔍 Deep Learning • Transfer Learning • Medical Imaging • CNN • Healthcare AI

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Library-Keras-red?logo=keras)
![CNN](https://img.shields.io/badge/Architecture-CNN-yellow)
![TransferLearning](https://img.shields.io/badge/Method-Transfer%20Learning-green)
![Domain](https://img.shields.io/badge/Domain-Healthcare-purple)

---

## 📘 Overview
**Multi-Class COVID-19 Detection from Chest X-ray Images** is a deep learning healthcare project that classifies chest X-ray images into **three diagnostic categories**:

- 🦠 **COVID-19**
- 🌫️ **Viral Pneumonia**
- 💨 **Normal (Healthy)**

Using **Convolutional Neural Networks (CNNs)** and **transfer learning** models like **ResNet-50, VGG-16, and DenseNet-121**, this system demonstrates how AI can assist radiologists in **rapid diagnosis, triage support, and pandemic-scale screening**.

---

## 🎯 Objective
Develop a **fully automated medical imaging pipeline** that:
- Classifies X-ray scans into **COVID-19 / Viral Pneumonia / Normal**
- Performs well on **imbalanced real-world medical datasets**
- Supports **Grad-CAM visual explanations** for interpretability
- Is deployable as a **web application or API**

---

## 💼 Business & Healthcare Use Cases
| Sector | Impact |
|--------|--------|
| 🏥 Hospitals | Reduce workload for radiologists & enable faster triage |
| 🌍 Remote Healthcare | Assist low-resource clinics with AI diagnostic support |
| 🧪 Public Health Screening | Enable scalable population-level testing during pandemics |
| 🎓 Medical Education | Train radiology learners using AI-explained visualizations |

---

## 🧠 Skills & Technologies Demonstrated
- Medical image preprocessing & augmentation
- Deep learning on radiographic images
- Fine-tuning + transfer learning (ResNet-50, VGG-16, DenseNet-121)
- Metrics analysis: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Grad-CAM interpretability for clinical trust
- Deployment using **Streamlit / Flask**

---

## ⚙️ Approach Summary

### 🔹 Data Preparation
- Visualized class distribution
- Handled imbalance via augmentation (rotation, flipping, brightness jitter)
- Resized + normalized images for CNN-compatibility

### 🔹 Model Development
- Baseline CNN model
- Transfer learning + fine-tuning with:
  - **ResNet-50**
  - **VGG-16**
  - **DenseNet-121**

### 🔹 Model Evaluation
- Performance metrics: **Accuracy, Precision, Recall, F1-Score, ROC-AUC**
- Diagnostic plots:
  - Training vs Validation Curves
  - Confusion Matrix
  - Class-wise ROC Curves

### 🔹 Deployment
- Web app built using **Streamlit**
- REST inference via **Flask API (optional)**
- **Grad-CAM heatmaps** to highlight disease-affected regions

---


<summary>📸 Click to view Streamlit UI screenshots</summary>

#### Home Page  
![Home Page](https://github.com/user-attachments/assets/2def778e-0a6b-4315-9b55-a41ef91af701)


#### Results Page  
![Result Page](https://github.com/user-attachments/assets/08b7dbfb-7749-4720-849e-ff1fbfeba079)

                        

---

## 🧩 Project Structure
```bash
COVID19_Xray_Classification/
│
├── multiclass-covid19.ipynb      # Model training & evaluation notebook
│
├── app.py                        # Streamlit web application
│
└── requirements.txt              # Project dependencies
```

---

## 🛠 Run Locally
Install dependencies:
```
pip install -r requirements.txt
```

Launch the application:
```
streamlit run app.py
```

