# 👕 Fashion MNIST Image Classifier (CNN + Streamlit)

An end-to-end Computer Vision and Deep Learning web application built using **TensorFlow/Keras** and **Streamlit**. The app classifies user-uploaded clothing images into one of 10 Fashion MNIST categories in real-time.

---

## 🌟 Key Features
- **CNN Architecture**: 2D Convolutional Neural Network trained on 60,000 Fashion MNIST images (~89.3% Test Accuracy).
- **Interactive Web App**: Modern Streamlit frontend for uploading images and viewing live predictions.
- **Smart Image Preprocessing**: Automatic background color detection and inversion (normalizes light/dark photo backgrounds to match model input specifications).
- **Confidence Breakdown**: Displays predicted class confidence along with full probability distribution across all 10 categories.

---

## 📂 Project Structure
```text
fashion-mnist-classifier/
├── app.py                            # Streamlit web application deployment script
├── cnn_algorithm_fashion_mnist.ipynb # CNN model training, evaluation, and plotting notebook
├── streamlit_demo.ipynb              # Streamlit deployment notebook version (demo/reference)
├── create_samples.py                 # Script to generate sample test images from dataset
├── fashion_mnist_cnn.keras           # Trained Keras model weights (add via Git LFS if >50MB)
├── requirements.txt                  # Python dependency requirements
└── sample_images/                    # Pre-extracted test sample images (git-ignored)
```

---

## 🏷️ Supported Categories
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

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

### 3. Run the Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your web browser.

---

## 📊 Model Performance
- **Training Epochs**: 5
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Validation Accuracy**: ~89.8%
- **Test Accuracy**: ~89.3%

---

## 🛠️ Tech Stack
- **Deep Learning Framework**: TensorFlow / Keras
- **Frontend Framework**: Streamlit
- **Data & Image Processing**: NumPy, Pillow
- **Data Visualization**: Matplotlib
