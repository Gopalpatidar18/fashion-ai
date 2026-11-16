"""Train a simple CNN on Fashion-MNIST and save fashion_model.keras."""
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fashion_model.keras"

print("Loading Fashion-MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values and add the grayscale channel dimension.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., None]
x_test = x_test[..., None]

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training CNN...")
model.fit(
    x_train, y_train,
    epochs=8,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy * 100:.2f}%")

model.save(MODEL_PATH)
print(f"Saved trained model to: {MODEL_PATH}")
