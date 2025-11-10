# 🩺 Multi-class COVID-19 Detection from Chest X-ray Images

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Library-Keras-red?logo=keras)
![CNN](https://img.shields.io/badge/Architecture-CNN-yellow)
![TransferLearning](https://img.shields.io/badge/Method-Transfer%20Learning-green)
![Domain](https://img.shields.io/badge/Domain-Healthcare-purple)

---

## 📘 Overview
**Multi-class COVID-19 Detection from Chest X-ray Images** is a deep learning–based diagnostic project that classifies chest X-rays into **three categories**:  
- 🦠 COVID-19  
- 🌫️ Viral Pneumonia  
- 💨 Normal  

Using **Convolutional Neural Networks (CNNs)** and **transfer learning** (ResNet-50, VGG-16, DenseNet), this project demonstrates how **AI in healthcare** can assist radiologists and healthcare providers in rapid diagnosis and large-scale screening.

The workflow covers **data preprocessing**, **model training**, **evaluation**, and **deployment** (as a web app or API).

---

## 🎯 Problem Statement
Rapid and accurate detection of **COVID-19** using chest X-ray images is critical to improve early diagnosis and reduce healthcare burden.  

The goal is to build an **end-to-end multi-class classification system** capable of identifying whether an X-ray belongs to:
- A patient infected with **COVID-19**
- A patient with **Viral Pneumonia**
- A **Normal** (healthy) individual

The model should provide interpretable and reliable results, suitable for use in both **clinical** and **remote healthcare settings**.

---

## 💼 Business Use Cases

### 1. 🏥 Clinical Support
- Assist radiologists by providing rapid triage support  
- Reduce diagnostic workload and time-to-diagnosis  

### 2. 🌍 Remote Healthcare
- Enable AI-assisted diagnostics in under-resourced or rural areas  
- Support telemedicine and mobile diagnostic platforms  

### 3. 🧪 Public Health Screening
- Conduct large-scale screening efficiently using automated detection  
- Provide preliminary analysis during pandemic surges  

### 4. 🎓 Training Tools
- Serve as an educational aid for radiology students  
- Demonstrate AI integration in medical imaging courses  

---

## 🧠 Skills Takeaway
- 🩻 **Medical Image Preprocessing & Augmentation**
- 🧠 **Deep Learning Architectures (CNNs)**
- 🔁 **Transfer Learning with ResNet-50, VGG-16, DenseNet**
- 🎯 **Multi-class Classification (COVID-19, Viral Pneumonia, Normal)**
- 📊 **Model Evaluation: Accuracy, ROC-AUC, Confusion Matrix**
- 🧪 **Experiment Tracking & Fine-tuning**
- ☁️ **End-to-End Workflow: Training → Evaluation → Deployment**

---

## ⚙️ Approach Summary

### 🧩 Step 1: Data Exploration & Preprocessing
- Visualized dataset class distribution  
- Handled class imbalance using augmentation (rotation, flipping, brightness jitter)  
- Resized and normalized X-ray images for uniformity  

### 🧠 Step 2: Model Development
- Built baseline CNN model for initial evaluation  
- Implemented **transfer learning** with:
  - ResNet-50  
  - VGG-16  
  - DenseNet-121  
- Fine-tuned top layers for optimal accuracy and generalization  

### 📈 Step 3: Model Evaluation
- Metrics used:
  - **Accuracy**, **Precision**, **Recall**, **F1-score**, **ROC-AUC**
- Visualizations:
  - **Confusion Matrix**
  - **ROC Curves**
  - **Training vs. Validation Accuracy/Loss**

### ☁️ Step 4: Deployment
- Integrated model with **Streamlit** or **Flask API**
- Implemented **Grad-CAM visualization** for interpretability  
- Deployed model for real-time prediction and web-based interaction  

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
│
├── multiclass-covid19.ipynb       
│ 
├── app.py                 
│  
└── requirements.txt                            
