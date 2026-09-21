# 🩺 Breast Ultrasound Multi-Task Analysis

A deep learning application for **breast ultrasound image analysis** using a **multi-task U-Net architecture** with a **pretrained ResNet34 encoder**.

The model performs **two tasks simultaneously**:

* 🎯 **Lesion Segmentation**
* 🩻 **Breast Ultrasound Classification**

---

## 🚀 Live Demo & Resources

<p align="center">

<a href="https://breast-ultrasound-segmentation.streamlit.app/">
  <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</a>

<a href="https://www.kaggle.com/code/samoura/breast-ultrasound-segmentation-u-net-1">
  <img src="https://img.shields.io/badge/📓%20Kaggle%20Notebook-View%20Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
</a>

</p>

---

## ✨ Project Overview

This project uses a **multi-task learning approach**, allowing a single model to learn both:

1. **Where the lesion is located** in the ultrasound image.
2. **Which class the image belongs to.**

The model shares the same pretrained encoder between the two tasks and then uses separate task-specific heads.

---

## 🧠 Model Architecture

The model is built using:

* **ResNet34** encoder pretrained on **ImageNet**
* **U-Net decoder** for lesion segmentation
* **Classification head** for three-class classification
* **Adaptive Global Average Pooling** before classification

### Architecture

```text
                         🩻 Input Image
                              │
                              ▼
                    ┌───────────────────┐
                    │  ResNet34 Encoder │
                    │  ImageNet Weights │
                    └─────────┬─────────┘
                              │
                       Shared Features
                         ┌────┴────┐
                         ▼         ▼
                ┌────────────┐  ┌───────────────┐
                │ U-Net      │  │ Classification│
                │ Decoder    │  │ Head          │
                └─────┬──────┘  └───────┬───────┘
                      ▼                  ▼
               🩹 Lesion Mask       🏷️ 3 Classes
```

---

## 🏷️ Classification Classes

The model predicts one of three classes:

| Class            | Description       |
| ---------------- | ----------------- |
| 🟢 **Benign**    | Benign lesion     |
| 🔴 **Malignant** | Malignant lesion  |
| ⚪ **Normal**     | Normal ultrasound |

---

## 🎯 Model Tasks

### 1. Lesion Segmentation

The segmentation branch predicts a **binary lesion mask** indicating the location of the lesion in the ultrasound image.

The segmentation output is converted into a binary mask using a probability threshold of **0.5**.

### 2. Image Classification

The classification branch predicts the ultrasound image category:

```text
Benign
Malignant
Normal
```

The model outputs class probabilities using **Softmax**.

---

## 📥 Input

The application accepts:

* `.jpg`
* `.jpeg`
* `.png`

Input images are resized to:

```text
256 × 256
```

---

## 📤 Output

For each uploaded ultrasound image, the application provides:

* 🏷️ Predicted class
* 📊 Classification confidence
* 📈 Class probabilities
* 🎯 Predicted lesion segmentation mask

---

## 🛠️ Technologies

<p align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/U--Net-Deep%20Learning-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/ResNet34-ImageNet-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>

</p>

---

## 📂 Repository Structure

```text
breast-ultrasound-multitask/
│
├── app.py
├── requirements.txt
├── README.md
│
└── src/
    ├── model.py
    └── preprocessing.py
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd breast-ultrasound-multitask
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 🔬 Inference Pipeline

```text
Upload Ultrasound Image
          ↓
      Preprocessing
          ↓
    ResNet34 Encoder
          ↓
     Shared Features
       ↙          ↘
 Segmentation    Classification
      ↓               ↓
 Lesion Mask       Class + Probability
```

---

## 📊 Model Evaluation

The model was evaluated using metrics appropriate for both tasks.

### Segmentation

* **Dice Score**
* **IoU (Intersection over Union)**

### Classification

* **Accuracy**
* **Class Probabilities**

---

## 📚 Dataset

The project uses the **Breast Ultrasound Images Dataset (BUSI)** for breast lesion segmentation and classification.

The dataset contains ultrasound images belonging to:

* Benign
* Malignant
* Normal

---

## 👩‍💻 Author

**Samira Hassan**

Computer Science Graduate | Machine Learning & Deep Learning

---

## 🔗 Project Links

🚀 **Live Application:**
https://breast-ultrasound-segmentation.streamlit.app/

📓 **Kaggle Notebook:**
https://www.kaggle.com/code/samoura/breast-ultrasound-segmentation-u-net-1
