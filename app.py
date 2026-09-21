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
# Simple Styling
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: #0f766e;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #64748b;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #1e293b;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        margin-top: 35px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
    }

    </style>
    """,
    unsafe_allow_html=True
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
# Sidebar
# =========================================================

with st.sidebar:

    st.title("🩺 About")

    st.write(
        "This application uses a Multi-Task U-Net "
        "architecture for breast ultrasound analysis."
    )

    st.divider()

    st.subheader("Architecture")
    st.write("U-Net + ResNet34 Encoder")

    st.subheader("Input Size")
    st.write("256 × 256")

    st.subheader("Tasks")
    st.write("• Lesion Segmentation")
    st.write("• Image Classification")

    st.subheader("Classes")
    st.write("• Benign")
    st.write("• Malignant")
    st.write("• Normal")


# =========================================================
# Header
# =========================================================

st.markdown(
    '<div class="main-title">🩺 Breast Ultrasound AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Multi-task deep learning for breast lesion '
    'segmentation and classification.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# Upload Image
# =========================================================

st.markdown(
    '<div class="section-title">📤 Upload Ultrasound Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an ultrasound image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# Prediction
# =========================================================

if uploaded_file is not None:

    # =====================================================
    # Open Image
    # =====================================================

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # =====================================================
    # Input Image
    # =====================================================

    st.markdown(
        '<div class="section-title">🖼️ Input Image</div>',
        unsafe_allow_html=True
    )

    image_col, info_col = st.columns(
        [2, 1],
        gap="large"
    )

    with image_col:

        st.image(
            image,
            caption="Uploaded Ultrasound Image",
            use_container_width=True
        )

    with info_col:

        st.subheader("Image Information")

        st.write(
            f"**Format:** "
            f"{image.format or 'Image'}"
        )

        st.write(
            f"**Original Size:** "
            f"{image.width} × {image.height}"
        )

        st.write(
            "**Model Input:** 256 × 256"
        )


    # =====================================================
    # Preprocessing
    # =====================================================

    input_tensor = preprocess_image(
        image
    )

    input_tensor = input_tensor.to(
        device
    )


    # =====================================================
    # Model Prediction
    # =====================================================

    with st.spinner(
        "Analyzing ultrasound image..."
    ):

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
    # Prediction Results
    # =====================================================

    st.markdown(
        '<div class="section-title">🎯 Prediction Results</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns(
        2,
        gap="large"
    )


    # =====================================================
    # Classification Result
    # =====================================================

    with result_col1:

        st.subheader("Classification")

        st.metric(
            label="Predicted Class",
            value=predicted_label
        )

        st.metric(
            label="Confidence",
            value=f"{confidence * 100:.2f}%"
        )


    # =====================================================
    # Segmentation Result
    # =====================================================

    with result_col2:

        st.subheader("Lesion Segmentation")

        st.image(
            predicted_mask,
            caption="Predicted Lesion Mask",
            use_container_width=True
        )


    # =====================================================
    # Classification Probabilities
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Classification Probabilities'
        '</div>',
        unsafe_allow_html=True
    )

    probability_col1, probability_col2, probability_col3 = st.columns(
        3
    )

    probability_columns = [
        probability_col1,
        probability_col2,
        probability_col3
    ]

    for class_id, class_name in class_names.items():

        probability = probabilities[
            0,
            class_id
        ].item()

        with probability_columns[class_id]:

            st.metric(
                label=class_name,
                value=f"{probability * 100:.2f}%"
            )

            st.progress(
                probability
            )


# =========================================================
# Disclaimer
# =========================================================

st.warning(
    "⚠️ **Research & Educational Use Only**\n\n"
    "This application is not a medical diagnostic tool "
    "and should not replace evaluation by a qualified "
    "healthcare professional."
)


# =========================================================
# Footer
# =========================================================

st.markdown(
    """
    <div class="footer">
        🩺 Breast Ultrasound AI
        <br>
        PyTorch • U-Net • ResNet34 • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
