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
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# Custom CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- Main App ---------- */

    .stApp {
        background: #f7f9fc;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }


    /* ---------- Header ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            #0f766e 0%,
            #155e75 100%
        );

        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;

        box-shadow:
            0 10px 30px rgba(15, 118, 110, 0.15);
    }

    .hero-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 750;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        color: rgba(255,255,255,0.88);
        font-size: 1.05rem;
        margin-bottom: 0;
        line-height: 1.6;
    }


    /* ---------- Section Titles ---------- */

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1f2937;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }


    /* ---------- Upload Box ---------- */

    .upload-info {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;

        box-shadow:
            0 4px 15px rgba(15, 23, 42, 0.04);
    }

    .upload-title {
        font-size: 1.05rem;
        font-weight: 650;
        color: #111827;
    }

    .upload-description {
        color: #6b7280;
        font-size: 0.92rem;
        margin-top: 0.35rem;
    }


    /* ---------- Result Cards ---------- */

    .result-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1.4rem;

        box-shadow:
            0 5px 20px rgba(15, 23, 42, 0.05);

        height: 100%;
    }

    .result-label {
        color: #6b7280;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }

    .result-value {
        color: #111827;
        font-size: 1.7rem;
        font-weight: 750;
    }


    /* ---------- Classification Colors ---------- */

    .benign {
        color: #047857;
    }

    .malignant {
        color: #dc2626;
    }

    .normal {
        color: #2563eb;
    }


    /* ---------- Probability Cards ---------- */

    .prob-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;

        box-shadow:
            0 3px 12px rgba(15, 23, 42, 0.035);
    }

    .prob-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.55rem;
    }

    .prob-name {
        font-weight: 650;
        color: #374151;
    }

    .prob-value {
        font-weight: 700;
        color: #111827;
    }


    /* ---------- Info Cards ---------- */

    .info-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.2rem;
        height: 100%;
    }

    .info-title {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 0.35rem;
    }

    .info-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #111827;
    }


    /* ---------- Disclaimer ---------- */

    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        color: #9a3412;
        font-size: 0.86rem;
        line-height: 1.5;
        margin-top: 2rem;
    }


    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.82rem;
        margin-top: 2.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e5e7eb;
    }


    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    .sidebar-title {
        font-size: 1.15rem;
        font-weight: 750;
        color: #111827;
    }

    .sidebar-text {
        color: #6b7280;
        font-size: 0.9rem;
        line-height: 1.6;
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
# Sidebar
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🩺 About the Model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-text">

        <br>

        This application uses a <b>Multi-Task U-Net</b>
        architecture for breast ultrasound analysis.

        <br><br>

        <b>Encoder</b><br>
        ResNet34 pretrained on ImageNet

        <br><br>

        <b>Tasks</b><br>
        • Lesion segmentation<br>
        • Three-class classification

        <br><br>

        <b>Input Size</b><br>
        256 × 256

        <br><br>

        <b>Classes</b><br>
        • Benign<br>
        • Malignant<br>
        • Normal

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.caption(
        "Research / educational application"
    )


# =========================================================
# Hero Header
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🩺 Breast Ultrasound AI
        </div>

        <div class="hero-subtitle">
            Multi-task deep learning for breast lesion
            segmentation and classification.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Upload Section
# =========================================================

st.markdown(
    '<div class="section-title">📤 Upload Ultrasound Image</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="upload-info">

        <div class="upload-title">
            Upload a breast ultrasound image
        </div>

        <div class="upload-description">
            Supported formats: JPG, JPEG, PNG
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# =========================================================
# Prediction
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # =====================================================
    # Image Preview
    # =====================================================

    st.markdown(
        '<div class="section-title">🖼️ Image Analysis</div>',
        unsafe_allow_html=True
    )

    image_col, info_col = st.columns(
        [1.5, 1],
        gap="large"
    )


    with image_col:

        st.image(
            image,
            caption="Uploaded Ultrasound Image",
            use_container_width=True
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
    # Quick Result
    # =====================================================

    with info_col:

        if predicted_label == "Benign":
            label_class = "benign"
            icon = "🟢"

        elif predicted_label == "Malignant":
            label_class = "malignant"
            icon = "🔴"

        else:
            label_class = "normal"
            icon = "🔵"


        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Predicted Class
                </div>

                <div class="result-value {label_class}">
                    {icon} {predicted_label}
                </div>

                <br>

                <div class="result-label">
                    Confidence
                </div>

                <div class="result-value">
                    {confidence * 100:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# Results
# =========================================================

    st.markdown(
        '<div class="section-title">🎯 Prediction Results</div>',
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
            <div class="result-card">

                <div class="result-label">
                    Segmentation
                </div>

                <div class="result-value">
                    🎯 Lesion Mask
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
    # Classification Probabilities
    # -----------------------------------------------------

    with result_col2:

        st.markdown(
            """
            <div class="result-card">

                <div class="result-label">
                    Classification
                </div>

                <div class="result-value">
                    📊 Class Probabilities
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        for class_id, class_name in class_names.items():

            probability = probabilities[
                0,
                class_id
            ].item()

            st.markdown(
                f"""
                <div class="prob-card">

                    <div class="prob-header">

                        <span class="prob-name">
                            {class_name}
                        </span>

                        <span class="prob-value">
                            {probability * 100:.2f}%
                        </span>

                    </div>

                </div>
                """,
                unsafe_allow_html=True
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

        ⚠️ <b>Important:</b>
        This application is intended for research and
        educational purposes only. It is not a medical
        diagnostic tool and should not be used as a substitute
        for professional medical evaluation.

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

        🩺 Breast Ultrasound Multi-Task Analysis
        <br>
        Built with PyTorch • U-Net • ResNet34 • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)
