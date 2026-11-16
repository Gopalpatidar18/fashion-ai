# Fashion Image Classification — CNN + Flask Dashboard

A beginner-friendly full-stack Deep Learning project. It trains a CNN on Fashion-MNIST and provides a Flask dashboard where a user can upload an image and see the predicted clothing category and confidence.

## Requirements
- Python 3.10, 3.11, or 3.12
- Internet connection for the first training run (to download Fashion-MNIST)
- About 1–2 GB free disk space is recommended for Python packages and training files

> Fashion-MNIST contains small, centered, grayscale 28×28 images. This starter project is best demonstrated with Fashion-MNIST-like images. Arbitrary real-world clothing photos may be classified incorrectly because they differ from the training data.

## Run on macOS

Install Python 3.10, 3.11, or 3.12, then open Terminal in the project folder and run:

```bash
bash run.sh
```

The script creates a virtual environment, installs dependencies, downloads Fashion-MNIST and trains the model on the first run, then starts the dashboard. The initial setup and training can take several minutes. Later runs reuse the environment and saved model.

## Open the dashboard
Visit:

http://127.0.0.1:5001

Upload a PNG/JPG image and click **Classify Image**.

If you want to use port 5000 instead, run:

```bash
PORT=5000 python app.py
```

## Run on Windows

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

## Model classes
1. T-shirt / Top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle Boot

## Project files
- `train_model.py` — trains and saves the CNN model
- `app.py` — Flask backend and image prediction route
- `templates/` — dashboard and prediction result pages
- `static/css/style.css` — dashboard styling
- `static/uploads/` — uploaded images

## Troubleshooting
- **Model file not found:** run `python train_model.py` before starting the Flask app.
- **TensorFlow install error:** use Python 3.11 in a fresh virtual environment, then reinstall requirements.
- **Port already in use:** change the final line in `app.py` to `app.run(port=5001, debug=True)` and open port 5001.
- **Poor predictions on phone photos:** this is expected with Fashion-MNIST. Use centered grayscale clothing images similar to the dataset, or upgrade the project to a larger real-world fashion dataset.
