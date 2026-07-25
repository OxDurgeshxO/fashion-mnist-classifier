# 👗 Fashion MNIST Image Classifier (CNN + Streamlit)

An end-to-end Computer Vision and Deep Learning web application built using **TensorFlow/Keras** and **Streamlit**. The app classifies user-uploaded clothing images into one of 10 Fashion MNIST categories in real-time.

> **⚡ Train the model first:** Run `python train_model.py` before launching the app. The trained `.keras` file is excluded from the repo due to size — train it locally or via the Colab notebook below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OxDurgeshxO/fashion-mnist-classifier/blob/main/cnn_algorithm_fashion_mnist.ipynb)

---

## 🌟 Key Features
- **CNN Architecture**: Deep 3-block Convolutional Neural Network with BatchNorm, Dropout, and data augmentation — trained on 60,000 Fashion MNIST images (~91%+ Test Accuracy)
- **Interactive Web App**: Streamlit frontend with Plotly probability charts, sidebar model info, and emoji-enhanced predictions
- **Smart Image Preprocessing**: Automatic background detection and inversion to normalize real-world photo backgrounds to match model input specification
- **Confidence Breakdown**: Plotly horizontal bar chart with color-highlighted predicted class and percentage labels across all 10 categories
- **Prediction Caching**: MD5 hash-based caching avoids re-inference on repeated Streamlit interactions

---

## 📂 Project Structure
```text
fashion-mnist-classifier/
├── app.py                            # Streamlit web application
├── train_model.py                    # Standalone training script (run this first)
├── cnn_algorithm_fashion_mnist.ipynb # CNN model training notebook (Colab-ready)
├── streamlit_demo.ipynb              # Streamlit deployment demo notebook
├── create_samples.py                 # Script to generate sample test images
├── requirements.txt                  # Python dependencies
├── .streamlit/config.toml            # Streamlit theme configuration
└── sample_images/                    # Pre-extracted test samples (git-ignored)
```

---

## 🧠 CNN Architecture

| Layer | Type | Details |
|---|---|---|
| 1 | Input | 28×28×1 grayscale |
| 2 | Conv2D | 32 filters, 3×3, ReLU, same padding |
| 3 | BatchNormalization | — |
| 4 | MaxPooling2D | 2×2 |
| 5 | Conv2D | 64 filters, 3×3, ReLU, same padding |
| 6 | BatchNormalization | — |
| 7 | MaxPooling2D | 2×2 |
| 8 | Conv2D | 128 filters, 3×3, ReLU, same padding |
| 9 | BatchNormalization | — |
| 10 | Flatten | — |
| 11 | Dense | 256 units, ReLU |
| 12 | Dropout | 0.5 |
| 13 | Output | 10 units, Softmax |

---

## 🏷️ Supported Categories

| # | Category | Emoji |
|---|---|---|
| 0 | T-shirt/top | 👕 |
| 1 | Trouser | 👖 |
| 2 | Pullover | 🧥 |
| 3 | Dress | 👗 |
| 4 | Coat | 🥼 |
| 5 | Sandal | 👡 |
| 6 | Shirt | 👔 |
| 7 | Sneaker | 👟 |
| 8 | Bag | 👜 |
| 9 | Ankle boot | 👢 |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Training Epochs | Up to 20 (EarlyStopping) |
| Optimizer | Adam + ReduceLROnPlateau |
| Loss Function | Categorical Crossentropy |
| Data Augmentation | Rotation, Shift, Zoom, Flip |
| Validation Accuracy | ~91%+ |
| Test Accuracy | ~91%+ |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/OxDurgeshxO/fashion-mnist-classifier.git
cd fashion-mnist-classifier
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python train_model.py
```
This downloads the Fashion MNIST dataset automatically, trains the CNN, and saves `fashion_mnist_cnn.keras` to the project directory.

### 4. Run the Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## ☁️ Deploy to Streamlit Cloud

1. Fork this repo
2. Add `fashion_mnist_cnn.keras` to the repo root (train locally and commit, or use Git LFS)
3. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub, and deploy `app.py`

> **Note:** The `.keras` model file is git-ignored by default. For Streamlit Cloud deployment, either commit the file directly (if <50MB) or use [Git LFS](https://git-lfs.github.com/).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Frontend | Streamlit |
| Charts | Plotly |
| Image Processing | NumPy, Pillow |
| Data Augmentation | Keras ImageDataGenerator |
