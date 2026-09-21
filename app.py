
import streamlit as st
import torch
from PIL import Image
import os
import gdown

from src.model import MultiTaskUNet
from src.preprocessing import preprocess_image


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Breast Ultrasound AI",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# Custom CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Main container */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .header {
        background: linear-gradient(135deg, #0f766e, #155e75);
        padding: 32px;
        border-radius: 18px;
        margin-bottom: 28px;
    }

    .header h1 {
        color: white;
        margin: 0;
        font-size: 38px;
    }

    .header p {
        color: #e0f2fe;
        font-size: 17px;
        margin-top: 10px;
        margin-bottom: 0;
    }

    /* Cards */
    .card {
        background: white;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    .card-title {
        color: #374151;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .card-value {
        color: #111827;
        font-size: 28px;
        font-weight: 700;
        margin-top: 6px;
    }

    /* Section title */
    .section-title {
        color: #111827;
        font-size: 22px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        padding: 15px 18px;
        border-radius: 12px;
        color: #9a3412;
        font-size: 14px;
        margin-top: 30px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 13px;
        margin-top: 35px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
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

    download_model()

    model = MultiTaskUNet()

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
# Header
# =========================================================

st.markdown(
    """
    <div class="header">

        <h1>🩺 Breast Ultrasound AI</h1>

        <p>
            Multi-task deep learning for breast lesion
            segmentation and classification.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.header("🩺 About")

    st.write(
        """
        This application uses a Multi-Task U-Net
        architecture for breast ultrasound analysis.
        """
    )

    st.divider()

    st.write("**Architecture**")
    st.write("U-Net + ResNet34 Encoder")

    st.write("**Input Size**")
    st.write("256 × 256")

    st.write("**Tasks**")
    st.write("• Lesion Segmentation")
    st.write("• Image Classification")

    st.write("**Classes**")
    st.write("• Benign")
    st.write("• Malignant")
    st.write("• Normal")


# =========================================================
# Upload Section
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
            caption="Uploaded Ultrasound",
            use_container_width=True
        )

    with info_col:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    Image Information
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(
            f"**Format:** {image.format or 'Image'}"
        )

        st.write(
            f"**Original Size:** {image.width} × {image.height}"
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
    # Prediction
    # =====================================================

    with st.spinner("Analyzing ultrasound image..."):

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
    # Prediction Summary
    # =====================================================

    st.markdown(
        '<div class="section-title">🎯 Prediction Summary</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    Classification
                </div>

                <div class="card-value">
                    {predicted_label}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    Confidence
                </div>

                <div class="card-value">
                    {confidence * 100:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # Results
    # =====================================================

    st.markdown(
        '<div class="section-title">🔬 Analysis Results</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns(
        2,
        gap="large"
    )


    # -----------------------------------------------------
    # Segmentation
    # -----------------------------------------------------

    with result_col1:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    Lesion Segmentation
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(
            predicted_mask,
            caption="Predicted Lesion Mask",
            use_container_width=True
        )


    # -----------------------------------------------------
    # Probabilities
    # -----------------------------------------------------

    with result_col2:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    Classification Probabilities
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        for class_id, class_name in class_names.items():

            probability = probabilities[
                0,
                class_id
            ].item()

            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                probability
            )


# =========================================================
# Disclaimer
# =========================================================

st.markdown(
    """
    <div class="disclaimer">

        ⚠️ <b>Research & Educational Use Only</b><br>

        This application is not a medical diagnostic tool
        and should not replace evaluation by a qualified
        healthcare professional.

    </div>
    """,
    unsafe_allow_html=True
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
