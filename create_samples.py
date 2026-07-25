import os
import tensorflow as tf
import numpy as np
from PIL import Image

# Use relative path so it works on any machine
output_dir = "sample_images"
os.makedirs(output_dir, exist_ok=True)

(x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()

class_names = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"
]

for i, name in enumerate(class_names):
    idx = np.where(y_train == i)[0][0]
    img = Image.fromarray(x_train[idx]).resize((280, 280), Image.NEAREST)
    img_path = os.path.join(output_dir, f"{name}.png")
    img.save(img_path)
    print(f"Saved: {img_path}")

print("All samples created successfully!")
