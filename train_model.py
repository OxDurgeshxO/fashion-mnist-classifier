"""train_model.py

Standalone script to train the Fashion MNIST CNN and save the model.
Run this once before launching the Streamlit app:

    python train_model.py

Outputs: fashion_mnist_cnn.keras
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(42)
np.random.seed(42)

NUM_CLASSES = 10
EPOCHS = 20
BATCH_SIZE = 64
MODEL_PATH = "fashion_mnist_cnn.keras"

# ── Load & preprocess ────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, -1)   # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)     # (10000, 28, 28, 1)

y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# ── Data augmentation ────────────────────────────────────────────────────────
# Mild augmentation suitable for Fashion MNIST (small 28x28 images)
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)
datagen.fit(x_train)

train_gen = datagen.flow(x_train, y_train_cat, batch_size=BATCH_SIZE, subset="training", seed=42)
val_gen = datagen.flow(x_train, y_train_cat, batch_size=BATCH_SIZE, subset="validation", seed=42)

# ── Model architecture ───────────────────────────────────────────────────────
# CNN Architecture:
# Input (28x28x1)
# └─ Conv2D(32, 3x3, relu) -> MaxPool(2x2)
# └─ Conv2D(64, 3x3, relu) -> MaxPool(2x2)
# └─ Conv2D(128, 3x3, relu)
# └─ Flatten
# └─ Dense(256, relu) -> Dropout(0.5)
# └─ Dense(10, softmax)
model = models.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ── Callbacks ────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
]

# ── Train ────────────────────────────────────────────────────────────────────
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# ── Evaluate ─────────────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test Loss     : {test_loss:.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
model.save(MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")
