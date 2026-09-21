🩺 **Breast Ultrasound AI | Multi-Task Deep Learning**

I’m excited to share my latest Computer Vision project, focused on **Breast Ultrasound Analysis** using a Multi-Task Deep Learning approach.

The project combines two complementary tasks within a single model:

• **Lesion Segmentation** — identifying and localizing the lesion at the pixel level.
• **Image Classification** — classifying ultrasound images into **Benign, Malignant, or Normal**.

🧠 **Model Architecture**

The model is built around a **U-Net architecture with a pretrained ResNet34 encoder**.

The shared encoder extracts visual features from the ultrasound image, which are then used for two tasks:

**Input Image → ResNet34 Encoder →**
→ U-Net Decoder → Lesion Mask
→ Classification Head → Class Probabilities

⚙️ **Technologies**

• PyTorch
• U-Net
• ResNet34 Transfer Learning
• Medical Image Segmentation
• Image Classification
• Streamlit

I also developed an interactive **Streamlit web application** where users can upload an ultrasound image and receive the predicted lesion mask, classification result, and class probabilities.

This project allowed me to gain hands-on experience in **multi-task learning, transfer learning, medical image analysis, and deep learning deployment**.

🔗 **Live Demo:**
https://breast-ultrasound-segmentation.streamlit.app/

🔗 **GitHub Repository:**
https://github.com/Samira-Hassann/Breast-Ultrasound-Segmentation-U-Net

🔗 **Kaggle Notebook:**
https://www.kaggle.com/code/samoura/breast-ultrasound-segmentation-u-net-1

#ComputerVision #DeepLearning #MedicalImaging #PyTorch #UNet #ResNet34 #ImageSegmentation #ImageClassification #Streamlit #MachineLearning #AI
