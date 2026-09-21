import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import gdown

from src.model import MultiTaskUNet
from src.preprocessing import preprocess_image


# =========================================================
# Configuration
# =========================================================

st.set_page_config(
    page_title="Breast Ultrasound Analysis",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# Model Configuration
# =========================================================

MODEL_PATH = "models/best_multitask_model.pth"

DRIVE_FILE_ID = "1rVhTWQ2ROPUQOvhNGuNYQ0829rZNehfT"


# =========================================================
# Download Model
# =========================================================

def download_model():

    if os.path.exists(MODEL_PATH):
        return

    os.makedirs(
        "models",
        exist_ok=True
    )

    url = (
        f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    )

    gdown.download(
        url,
        MODEL_PATH,
        quiet=False
    )


# =========================================================
# Load Model
# =========================================================

@st.cache_resource
def load_model():

    # Download model if it doesn't exist
    download_model()

    # Create model architecture
    model = MultiTaskUNet()

    # Load trained weights
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model = model.to(device)

    model.eval()

    return model


model = load_model()


# =========================================================
# Class Names
# =========================================================

class_names = {
    0: "Benign",
    1: "Malignant",
    2: "Normal"
}


# =========================================================
# Title
# =========================================================

st.title("🩺 Breast Ultrasound Analysis")

st.write(
    """
    Upload a breast ultrasound image to obtain:
    
    - Lesion segmentation
    - Breast lesion classification
    """
)


# =========================================================
# Upload Image
# =========================================================

uploaded_file = st.file_uploader(
    "Upload an ultrasound image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# Prediction
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        width=500
    )


    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    input_tensor = preprocess_image(image)

    input_tensor = input_tensor.to(device)


    # -----------------------------------------------------
    # Model Prediction
    # -----------------------------------------------------

    with torch.no_grad():

        segmentation_output, classification_output = model(
            input_tensor
        )


    # =====================================================
    # Classification
    # =====================================================

    probabilities = torch.softmax(
        classification_output,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    predicted_label = class_names[
        predicted_class
    ]

    confidence = probabilities[
        0,
        predicted_class
    ].item()


    # =====================================================
    # Segmentation
    # =====================================================

    segmentation_probability = torch.sigmoid(
        segmentation_output
    )

    predicted_mask = (
        segmentation_probability > 0.5
    ).float()


    predicted_mask = (
        predicted_mask[0]
        .cpu()
        .squeeze(0)
        .numpy()
    )


    # =====================================================
    # Display Results
    # =====================================================

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Classification",
            predicted_label
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    with col2:

        st.write("Segmentation Mask")

        st.image(
            predicted_mask,
            width=400
        )


    # =====================================================
    # Classification Probabilities
    # =====================================================

    st.subheader("Class Probabilities")

    for class_id, class_name in class_names.items():

        probability = probabilities[
            0,
            class_id
        ].item()

        st.write(
            f"{class_name}: "
            f"{probability * 100:.2f}%"
        )

        st.progress(
            probability
        )
