import streamlit as st
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Breast Ultrasound Segmentation",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_model():

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    model.load_state_dict(
        torch.load(
            "best_unet.pth",
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    return model


model = load_model()


# ============================================================
# Image Preprocessing
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# ============================================================
# Prediction Function
# ============================================================

def predict(image):

    # Keep original image for display
    original_image = image.copy()

    # Convert image to RGB
    image = image.convert("RGB")

    # Same preprocessing used during training
    image_tensor = image_transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move to device
    image_tensor = image_tensor.to(device)

    # Prediction
    with torch.no_grad():

        pred = model(image_tensor)

        prob = torch.sigmoid(pred)

        pred_mask = (prob > 0.5).float()

    # Convert prediction to NumPy
    mask = (
        pred_mask[0]
        .squeeze(0)
        .cpu()
        .numpy()
    )

    return original_image, mask


# ============================================================
# Title
# ============================================================

st.title("🩺 Breast Ultrasound Lesion Segmentation")

st.write(
    "Upload a breast ultrasound image and the model "
    "will segment the lesion."
)


# ============================================================
# Upload Image
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an ultrasound image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# Run Prediction
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    original_image, predicted_mask = predict(image)


    # ========================================================
    # Create Overlay
    # ========================================================

    original_array = np.array(
        original_image.resize((256, 256)).convert("RGB")
    )

    overlay = original_array.copy()

    # Highlight predicted lesion
    overlay[predicted_mask == 1] = [255, 0, 0]


    # Blend original image and mask
    blended = (
        0.6 * original_array +
        0.4 * overlay
    ).astype(np.uint8)


    # ========================================================
    # Display Results
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("Original Image")

        st.image(
            original_image,
            use_container_width=True
        )


    with col2:

        st.subheader("Predicted Mask")

        st.image(
            predicted_mask,
            clamp=True,
            use_container_width=True
        )


    with col3:

        st.subheader("Segmentation Overlay")

        st.image(
            blended,
            use_container_width=True
        )


    # ========================================================
    # Prediction Information
    # ========================================================

    st.success(
        "Segmentation completed successfully."
    )
