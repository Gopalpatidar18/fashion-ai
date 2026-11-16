import os
from pathlib import Path
import uuid
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_PATH = BASE_DIR / "fashion_model.keras"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = "change-this-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

CLASS_NAMES = [
    "T-shirt / Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]
model = None

def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model file not found. Run: python train_model.py"
            )
        model = load_model(MODEL_PATH)
    return model

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(path):
    image = Image.open(path).convert("L")
    # Fashion-MNIST expects 28x28 grayscale images. This simple fit works
    # best for centered clothing images similar to the training dataset.
    image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype="float32") / 255.0
    return array.reshape(1, 28, 28, 1)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("Please choose an image first.")
        return redirect(url_for("index"))

    file = request.files["image"]
    if not file or file.filename == "":
        flash("Please choose an image first.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload PNG, JPG, JPEG, BMP or WEBP.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    extension = Path(filename).suffix.lower()
    saved_name = f"{uuid.uuid4().hex}{extension}"
    saved_path = UPLOAD_DIR / saved_name
    file.save(saved_path)

    try:
        predictions = get_model().predict(prepare_image(saved_path), verbose=0)[0]
        best_index = int(np.argmax(predictions))
        confidence = float(predictions[best_index]) * 100
        ranked = sorted(
            zip(CLASS_NAMES, predictions.tolist()),
            key=lambda item: item[1],
            reverse=True
        )[:3]
        top_predictions = [
            {"name": name, "confidence": round(float(score) * 100, 2)}
            for name, score in ranked
        ]
        return render_template(
            "result.html",
            image_url=url_for("static", filename=f"uploads/{saved_name}"),
            predicted_class=CLASS_NAMES[best_index],
            confidence=round(confidence, 2),
            top_predictions=top_predictions
        )
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        flash(f"Prediction could not be completed: {exc}")
        return redirect(url_for("index"))

@app.errorhandler(413)
def too_large(_error):
    flash("Image is too large. Maximum upload size is 5 MB.")
    return redirect(url_for("index"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
