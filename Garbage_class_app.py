import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Set page config MUST be the first Streamlit command
st.set_page_config(
    page_title="Garbage Classifier ♻️",
    page_icon="🗑️",
    layout="centered",
)

# Mapping of class indices to labels with emojis
class_names = {
    0: "Cardboard 📦",
    1: "Glass 🍾",
    2: "Metal 🛠️",
    3: "Paper 📄",
    4: "Plastic 🧴",
    5: "Trash 🗑️"
}

@st.cache_resource
def load_model_from_path(model_path):
    return load_model(model_path)

model = load_model_from_path("C:/Users/Asus/Downloads/garbage_final_model.pth")

# Title and description with HTML styling
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>Garbage Classification ♻️</h1>
    <p style='text-align: center; font-size: 18px;'>
    Upload an image of garbage, and our AI model will classify it into one of six categories.
    </p>
    """,
    unsafe_allow_html=True
)

# Sidebar with instructions
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Click 'Browse files' and upload an image (jpg, png, jpeg).  
    2. The model expects images of size 244x244 pixels (resize is automatic).  
    3. After upload, the prediction will appear below with confidence scores.  
    4. Supported categories: Cardboard, Glass, Metal, Paper, Plastic, Trash.
    """)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess to model input size
        img = image.resize((244, 244))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict and get result
        prediction = model.predict(img)[0]
        pred_index = np.argmax(prediction)
        pred_label = class_names.get(pred_index, "Unknown")

        st.markdown(f"### Prediction: **{pred_label}**")
        st.write("Confidence scores:")
        for i, score in enumerate(prediction):
            label = class_names.get(i, f"Class {i}")
            st.write(f"- {label}: {score:.3f}")

    except Exception as e:
        st.error(f"Error processing image or predicting: {e}")
else:
    st.info("Please upload an image file to classify.")
