import os
import sys

import numpy as np
from PIL import Image

# Official Fashion MNIST class names — kept in sync with app.py CLASS_NAMES
CLASS_NAMES = [
    "T-shirt_top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"
]


def create_samples(output_dir: str = "sample_images") -> None:
    """Generate one sample PNG per Fashion MNIST class from the training set.

    Args:
        output_dir: Directory to save the sample images into.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        import tensorflow as tf
        (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
    except Exception as e:
        print(f"Failed to load Fashion MNIST dataset: {e}")
        print("Check your internet connection and TensorFlow installation.")
        sys.exit(1)

    for i, name in enumerate(CLASS_NAMES):
        idx = np.where(y_train == i)[0][0]
        # Use Image.Resampling.NEAREST (Pillow >= 9.1 compatible)
        img = Image.fromarray(x_train[idx]).resize((280, 280), Image.Resampling.NEAREST)
        img_path = os.path.join(output_dir, f"{name}.png")
        img.save(img_path)
        print(f"Saved: {img_path}")

    print("All samples created successfully!")


if __name__ == "__main__":
    create_samples()
