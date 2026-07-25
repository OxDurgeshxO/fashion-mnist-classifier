import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Page config
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    layout="centered"
)

# Load the saved Keras model
@st.cache_resource
def load_model():
    model_path = "fashion_mnist_cnn.keras"
    if not os.path.exists(model_path) and os.path.exists("fashion_mnist_cnn (1).keras"):
        model_path = "fashion_mnist_cnn (1).keras"
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}. Make sure 'fashion_mnist_cnn.keras' exists in the app directory.")
        st.stop()

model = load_model()

# Fashion MNIST class names (0-9)
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

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

    # Read and show image safely
    try:
        image = Image.open(uploaded_file).convert("L")  # grayscale
    except Exception as e:
        st.error(f"Could not read image file: {e}. Please upload a valid JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded image", width=200)

    # Preprocess: resize to 28x28, normalize
    img_resized = image.resize((28, 28))
    img_array = np.array(img_resized).astype("float32") / 255.0

    # Fashion MNIST models expect white/bright clothing on a dark/black background.
    # If the uploaded image has a light background (mean brightness > 0.5), invert colors.
    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array

    # Reshape for Keras model (1, 28, 28, 1)
    input_tensor = np.expand_dims(img_array, axis=-1)   # (28, 28, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0) # (1, 28, 28, 1)

    # Show preprocessed image preview
    with st.expander("View 28x28 Preprocessed Model Input"):
        st.image(img_array, caption="Preprocessed 28x28 input (bright object on dark background)", width=140)

    # Predict (verbose=0 suppresses console noise in Streamlit)
    predictions = model.predict(input_tensor, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index])

    # Display prediction
    st.subheader("Prediction")
    st.write(f"Class: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}")

    # Full probability distribution
    st.subheader("All class probabilities")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{class_names[i]}: {prob:.2f}")
