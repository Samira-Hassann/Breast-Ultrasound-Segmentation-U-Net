import streamlit as st

# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Breast Ultrasound AI",
    page_icon="🩺",
    layout="wide"
)

# =========================
# Simple Styling
# =========================

st.markdown("""
<style>

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1 {
    color: #0f766e;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

.footer {
    text-align: center;
    color: #888;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #ddd;
}

</style>
""", unsafe_allow_html=True)


# =========================
# Sidebar
# =========================

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


# =========================
# Header
# =========================

st.title("🩺 Breast Ultrasound AI")

st.write(
    "Multi-task deep learning for breast lesion "
    "segmentation and classification."
)

st.divider()


# =========================
# Upload
# =========================

st.markdown(
    '<div class="section-title">📤 Upload Ultrasound Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an ultrasound image",
    type=["jpg", "jpeg", "png"]
)


# =========================
# Main App
# =========================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown(
        '<div class="section-title">🖼️ Input Image</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Ultrasound Image",
            use_container_width=True
        )

    with col2:

        st.subheader("Image Information")

        st.write(f"**Format:** {image.format or 'Image'}")
        st.write(
            f"**Original Size:** "
            f"{image.width} × {image.height}"
        )
        st.write("**Model Input:** 256 × 256")

    # =========================
    # Preprocessing
    # =========================

    input_tensor = preprocess_image(image)
    input_tensor = input_tensor.to(device)

    # =========================
    # Prediction
    # =========================

    with st.spinner("Analyzing ultrasound image..."):

        with torch.no_grad():

            segmentation_output, classification_output = model(
                input_tensor
            )

    # =========================
    # Classification
    # =========================

    probabilities = torch.softmax(
        classification_output,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    predicted_label = class_names[predicted_class]

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    # =========================
    # Segmentation
    # =========================

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

    # =========================
    # Prediction Summary
    # =========================

    st.markdown(
        '<div class="section-title">🎯 Prediction Summary</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Classification",
            value=predicted_label
        )

    with col2:
        st.metric(
            label="Confidence",
            value=f"{confidence * 100:.2f}%"
        )

    # =========================
    # Results
    # =========================

    st.markdown(
        '<div class="section-title">🔬 Analysis Results</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Lesion Segmentation")

        st.image(
            predicted_mask,
            caption="Predicted Lesion Mask",
            use_container_width=True
        )

    with col2:

        st.subheader("Classification Probabilities")

        for class_id, class_name in class_names.items():

            probability = probabilities[
                0,
                class_id
            ].item()

            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(probability)


# =========================
# Disclaimer
# =========================

st.warning(
    "⚠️ Research & Educational Use Only\n\n"
    "This application is not a medical diagnostic tool "
    "and should not replace evaluation by a qualified "
    "healthcare professional."
)


# =========================
# Footer
# =========================

st.divider()

st.caption(
    "🩺 Breast Ultrasound AI • "
    "PyTorch • U-Net • ResNet34 • Streamlit"
)
