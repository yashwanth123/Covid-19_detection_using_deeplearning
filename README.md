# COVID-19 Detection Using Lung X-rays

This project explores the use of deep learning and transfer learning to classify chest X-ray images into **COVID-19**, **Pneumonia**, and **Normal** categories. It was developed as part of a final year B.Tech project at GITAM University.

---

## 🧠 Objective

To build an AI-based system for early and cost-effective detection of COVID-19 infections using lung X-ray images, especially in areas lacking access to RT-PCR or CT scan facilities.

---

## 📊 Dataset

We used publicly available datasets from:
- [Kaggle - Chest X-ray Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [GitHub - education454 COVID-19 Dataset](https://github.com/education454/datasets)
- [Qatar University - COVID-19 X-ray Images](#)

**Total images:** 6568  
- COVID-19: 712  
- Pneumonia: 4273  
- Normal: 1583  

---

## 🏗️ Project Architecture

The project has two phases:
1. **Image Preprocessing:** resizing, grayscale conversion, rescaling, augmentation (optional)
2. **Model Training & Evaluation:** using both custom CNN and pretrained models

### Classification Scenarios:
- **Binary:** COVID-19 vs Normal
- **Categorical:** COVID-19 vs Pneumonia vs Normal

---

## 🔍 Models Used

### Custom CNNs:
- 5 CNNs with varying convolutional layers and filter sizes

### Transfer Learning Models:
- VGG16
- DenseNet121 & DenseNet201
- ResNet50
- InceptionV3
- InceptionResNetV2

All models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC (ROC)

---

## 🧪 Key Results

### Binary Classification (Best):
| Model | Accuracy | F1-Score | AUC |
|-------|----------|----------|-----|
| **InceptionResNetV2** | 99.59% | 0.996 | 99.2 |

### Categorical Classification (Best):
| Model | Accuracy | F1-Score | AUC |
|-------|----------|----------|-----|
| **DenseNet121** | 79.0% | 0.82 | 92.0 |

### Larger Dataset (Handcrafted):
| Model | Accuracy (Test) | F1-Score | AUC |
|-------|------------------|----------|-----|
| **DenseNet201** | 83.6% | 0.81 | 95.6 |

---

## 🧑‍💻 Tech Stack

- **Language:** Python  
- **Libraries:** TensorFlow, Keras, OpenCV, Sklearn, NumPy, Matplotlib  
- **Web Framework:** Flask (for frontend and prediction interface)  
- **Frontend:** Responsive UI for mobile and tablet (with COVID stats, helper info, X-ray upload)

---

## 🧩 Features

- Responsive web UI to upload and predict from X-ray
- Backend with Flask API for inference
- Transfer learning with ImageNet weights
- Model evaluation using metrics and confusion matrices

---

## 🔬 Limitations

- Overfitting observed due to small dataset
- Bias in multiclass due to pneumonia-heavy data
- Need for more COVID-positive X-ray data

---

## 🚀 Future Work

- Improve classification with Ensemble Learning
- Fine-tune weights and freeze layers better
- Use high-resolution consistent PA X-ray images only
- Implement model explainability (e.g., Grad-CAM)
- Deploy as a web service or mobile app

---

## 👨‍🎓 Authors

- L. Naga Sai Sri Ravi Teja  
- S. Ritesh Dev  
- K. Bharath  
- **T. Yashwanth Sai**

Under the guidance of **Dr. Don S. Kumar**  
Dept. of Computer Science, GITAM University

---

