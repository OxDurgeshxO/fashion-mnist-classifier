import hashlib
import io
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image

# Page config
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👗",
    layout="centered"
)

# Fashion MNIST official class names (0-9)
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

CLASS_EMOJIS = ["👕", "👖", "🧥", "👗", "🥼", "👡", "👔", "👟", "👜", "👢"]


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the trained Keras CNN model from disk.

    Returns:
        tf.keras.Model or None if the model file is missing or fails to load.
    """
    model_path = "fashion_mnist_cnn.keras"
    if not os.path.exists(model_path):
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(
            f"Failed to load model: {e}. "
            "Make sure 'fashion_mnist_cnn.keras' exists in the app directory."
        )
        return None


# BUG-001 FIX: wrap top-level model call in try/except so app never crashes silently
try:
    model = load_model()
except Exception as e:
    st.error(f"Unexpected error while loading model: {e}")
    st.stop()

if model is None:
    st.error(
        "⚠️ Model file `fashion_mnist_cnn.keras` not found.\n\n"
        "Run `python train_model.py` first to train and save the model, "
        "then restart the app."
    )
    st.info("Or open the Colab notebook via the badge in README to train and download the model.")
    st.stop()

# App title and description
st.title("👗 Fashion MNIST Image Classifier")
st.write(
    "Upload a clothing image (JPG/PNG). "
    "The CNN model will resize it to 28×28 grayscale and predict the fashion category."
)

# Sidebar: model architecture info — BUG-004 FIX: synced with actual train_model.py architecture
with st.sidebar:
    st.header("🧠 Model Architecture")
    st.markdown("""
    | Layer | Details |
    |---|---|
    | Input | 28×28×1 grayscale |
    | Conv2D (1) | 32 filters, 3×3, ReLU + BatchNorm |
    | MaxPool2D | 2×2 |
    | Conv2D (2) | 64 filters, 3×3, ReLU + BatchNorm |
    | MaxPool2D | 2×2 |
    | Conv2D (3) | 128 filters, 3×3, ReLU + BatchNorm |
    | Flatten | — |
    | Dense | 256 units, ReLU + Dropout (0.5) |
    | Output | 10 units, Softmax |
    """)
    st.markdown("---")
    st.header("📊 Model Performance")
    st.metric("Test Accuracy", "~89.3%")
    st.metric("Validation Accuracy", "~89.8%")
    st.metric("Training Epochs", "15")
    st.metric("Optimizer", "Adam")

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

    # BUG-002 FIX: capture raw bytes BEFORE any seek/verify so getvalue() is always reliable
    raw_bytes = uploaded_file.getvalue()

    # Server-side image content validation via Pillow verify()
    try:
        _verify_img = Image.open(io.BytesIO(raw_bytes))
        _verify_img.verify()
    except Exception:
        st.error("Invalid or corrupted image file. Please upload a valid JPG or PNG.")
        st.stop()

    # Read image as grayscale
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("L")
    except Exception as e:
        st.error(f"Could not read image file: {e}. Please upload a valid JPG or PNG.")
        st.stop()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Uploaded image", width=180)

    # Preprocess: resize to 28x28, normalize
    img_resized = image.resize((28, 28))
    img_array = np.array(img_resized).astype("float32") / 255.0

    # Background detection using corner pixels.
    # Fashion MNIST expects bright clothing on a dark background.
    corners = np.concatenate([
        img_array[:3, :3].flatten(),
        img_array[:3, -3:].flatten(),
        img_array[-3:, :3].flatten(),
        img_array[-3:, -3:].flatten(),
    ])
    if corners.mean() > 0.5:
        img_array = 1.0 - img_array

    # Reshape for Keras model: (1, 28, 28, 1)
    input_tensor = np.expand_dims(img_array, axis=-1)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Show preprocessed image preview
    with col2:
        with st.expander("🔍 View 28×28 Preprocessed Model Input"):
            st.image(
                img_array,
                caption="Preprocessed 28×28 input (bright object on dark background)",
                width=140
            )

    # Cache predictions by image hash (BUG-002 FIX: uses pre-captured raw_bytes)
    img_hash = hashlib.md5(raw_bytes).hexdigest()
    if st.session_state.get("last_hash") != img_hash:
        with st.spinner("Classifying..."):
            predictions = model.predict(input_tensor, verbose=0)
        st.session_state["last_hash"] = img_hash
        st.session_state["predictions"] = predictions
    else:
        predictions = st.session_state["predictions"]

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = CLASS_NAMES[predicted_index]
    predicted_emoji = CLASS_EMOJIS[predicted_index]
    confidence = float(predictions[0][predicted_index])

    # Display prediction
    st.markdown("---")
    st.subheader("🎯 Prediction")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(label="Predicted Class", value=f"{predicted_emoji} {predicted_class}")
    with col_b:
        st.metric(label="Confidence", value=f"{confidence * 100:.1f}%")

    # Full probability distribution — Plotly horizontal bar chart
    st.subheader("📊 All Class Probabilities")
    probs = [float(predictions[0][i]) for i in range(len(CLASS_NAMES))]
    labels = [f"{CLASS_EMOJIS[i]} {CLASS_NAMES[i]}" for i in range(len(CLASS_NAMES))]
    colors = [
        "#4F86C6" if i != predicted_index else "#2ECC71"
        for i in range(len(CLASS_NAMES))
    ]

    fig = go.Figure(go.Bar(
        x=probs,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{p * 100:.1f}%" for p in probs],
        textposition="outside",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"
    ))
    fig.update_layout(
        xaxis=dict(title="Probability", range=[0, 1.15], tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
        height=420,
        margin=dict(l=10, r=60, t=20, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Multi-image note
    st.info("💡 **Tip:** To classify multiple images, upload them one at a time. Each upload is independently cached.")
