import hashlib
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Page config
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    layout="centered"
)

# Fashion MNIST official class names (0-9)
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


@st.cache_resource
def load_model():
    """Load the trained Keras CNN model from disk.

    Returns:
        tf.keras.Model or None if the model file is missing.
    """
    model_path = "fashion_mnist_cnn.keras"
    if not os.path.exists(model_path):
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}. Make sure 'fashion_mnist_cnn.keras' exists in the app directory.")
        return None


model = load_model()
if model is None:
    st.error("Model file 'fashion_mnist_cnn.keras' not found. Please add it to the app directory.")
    st.stop()

# App title and description
st.title("Fashion MNIST Image Classifier")
st.write(
    "Upload a clothing image (JPG/PNG). "
    "The model will resize it to 28x28 grayscale and predict the fashion category."
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # File size validation (5MB limit)
    if uploaded_file.size > 5 * 1024 * 1024:
        st.error("File too large. Please upload an image under 5MB.")
        st.stop()

    # Server-side image content validation via Pillow verify()
    try:
        _verify_img = Image.open(uploaded_file)
        _verify_img.verify()  # raises if file is not a valid image
        uploaded_file.seek(0)  # reset stream after verify()
    except Exception:
        st.error("Invalid or corrupted image file. Please upload a valid JPG or PNG.")
        st.stop()

    # Read image as grayscale
    try:
        image = Image.open(uploaded_file).convert("L")
    except Exception as e:
        st.error(f"Could not read image file: {e}. Please upload a valid JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded image", width=200)

    # Preprocess: resize to 28x28, normalize
    img_resized = image.resize((28, 28))
    img_array = np.array(img_resized).astype("float32") / 255.0

    # Background detection using corner pixels instead of full-image mean.
    # Fashion MNIST expects bright clothing on a dark background.
    # Sample the four 3x3 corners to estimate background brightness.
    corners = np.concatenate([
        img_array[:3, :3].flatten(),
        img_array[:3, -3:].flatten(),
        img_array[-3:, :3].flatten(),
        img_array[-3:, -3:].flatten(),
    ])
    if corners.mean() > 0.5:
        img_array = 1.0 - img_array

    # Reshape for Keras model: (1, 28, 28, 1)
    input_tensor = np.expand_dims(img_array, axis=-1)    # (28, 28, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, 28, 28, 1)

    # Show preprocessed image preview
    with st.expander("View 28x28 Preprocessed Model Input"):
        st.image(
            img_array,
            caption="Preprocessed 28x28 input (bright object on dark background)",
            width=140
        )

    # Cache predictions by image hash to avoid re-running on every Streamlit interaction
    img_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    if st.session_state.get("last_hash") != img_hash:
        with st.spinner("Classifying..."):
            predictions = model.predict(input_tensor, verbose=0)
        st.session_state["last_hash"] = img_hash
        st.session_state["predictions"] = predictions
    else:
        predictions = st.session_state["predictions"]

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index])

    # Display prediction — confidence as percentage
    st.subheader("Prediction")
    st.write(f"Class: **{predicted_class}**")
    st.write(f"Confidence: **{confidence * 100:.1f}%**")

    # Full probability distribution as bar chart
    st.subheader("All Class Probabilities")
    prob_dict = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
    st.bar_chart(prob_dict)
