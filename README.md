# Breast Ultrasound Multi-Task Analysis

A deep learning application for breast ultrasound image analysis using a pretrained U-Net architecture with a ResNet34 encoder.

The model performs two tasks simultaneously:

1. Lesion segmentation
2. Image classification

## Model Architecture

The model uses:

- ResNet34 encoder pretrained on ImageNet
- U-Net decoder for lesion segmentation
- Classification head for three classes

Classification classes:

- Benign
- Malignant
- Normal

Architecture:

                           Image

                             ↓

                     ResNet34 Encoder

                            ↓

                    Shared Features

                  ↙              ↘

           U-Net Decoder    Classification Head

                  ↓                ↓

             Segmentation     3 Classes


## Input

The application accepts:

- JPG
- JPEG
- PNG

Images are resized to:

256 × 256

## Output

The application provides:

- Predicted breast lesion class
- Classification confidence
- Class probabilities
- Predicted lesion segmentation mask

## Project Structure

``` breast-ultrasound-multitask/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── best_multitask_model.pth
│
└── src/
    ├── model.py
    └── preprocessing.py
