import os
import numpy as np
from PIL import Image

# Use relative path so it works on any machine
output_dir = "sample_images"
os.makedirs(output_dir, exist_ok=True)

try:
    import tensorflow as tf
    (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
except Exception as e:
    print(f"Failed to load Fashion MNIST dataset: {e}")
    print("Check your internet connection and TensorFlow installation.")
    exit(1)

class_names = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"
]

for i, name in enumerate(class_names):
    idx = np.where(y_train == i)[0][0]
    # Use Image.Resampling.NEAREST (Pillow >= 9.1 compatible)
    img = Image.fromarray(x_train[idx]).resize((280, 280), Image.Resampling.NEAREST)
    img_path = os.path.join(output_dir, f"{name}.png")
    img.save(img_path)
    print(f"Saved: {img_path}")

print("All samples created successfully!")
