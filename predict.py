"""
predict.py – Standalone inference for gender detection.

This module provides a clean, reusable API for predicting gender from a
single face image.  It can be imported by the Streamlit app or used from
the command line.

Usage (CLI):
    python predict.py path/to/face.jpg
    python predict.py --model models/gender_detection_model.h5 --image face.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from config import MODEL_FILE, INPUT_SHAPE, CLASS_NAMES


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(image_path: str | Path) -> np.ndarray:
    """Load a face image and prepare it for MobileNetV2 inference."""
    img = Image.open(str(image_path)).convert("RGB")
    img = img.resize((INPUT_SHAPE[0], INPUT_SHAPE[1]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Expand batch dimension → shape (1, 224, 224, 3)
    arr = np.expand_dims(arr, axis=0)
    return arr


# ── Model loading ─────────────────────────────────────────────────────────────
_model_cache = {}


def load_model(model_path: str | Path = MODEL_FILE) -> keras.Model:
    """Load the trained Keras model (cached after first call)."""
    model_path = str(model_path)
    if model_path not in _model_cache:
        print(f"[*] Loading model from {model_path} …")
        _model_cache[model_path] = keras.models.load_model(model_path)
    return _model_cache[model_path]


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_gender(
    image_path: str | Path,
    model_path: str | Path = MODEL_FILE,
) -> dict:
    """Predict gender from a single face image.

    Returns
    -------
    dict
        ``{"gender": str, "confidence": float, "elapsed_ms": float}``
    """
    model = load_model(model_path)

    # Validate file
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    # Preprocess
    batch = preprocess_image(image_path)

    # Inference
    start = time.perf_counter()
    preds = model.predict(batch)
    elapsed = (time.perf_counter() - start) * 1000

    # Decode
    probs = preds[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx] * 100)

    return {
        "gender": CLASS_NAMES[idx],
        "confidence": round(confidence, 2),
        "probabilities": {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASS_NAMES))},
        "elapsed_ms": round(elapsed, 2),
    }


# ── Batch prediction ──────────────────────────────────────────────────────────
def predict_batch(
    image_paths: list[str | Path],
    model_path: str | Path = MODEL_FILE,
) -> list[dict]:
    """Run prediction on a list of images."""
    results = []
    for path in image_paths:
        try:
            result = predict_gender(path, model_path)
            result["image"] = str(path)
            results.append(result)
        except Exception as exc:
            results.append({"image": str(path), "error": str(exc)})
    return results


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict gender from a face image.")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to the face image.")
    parser.add_argument("--model", "-m", type=str, default=str(MODEL_FILE), help="Path to the .h5 model file.")
    args = parser.parse_args()

    print("=" * 50)
    print("  Gender Detection – Prediction")
    print("=" * 50)

    try:
        result = predict_gender(args.image, args.model)
        print(f"\n  Image      : {args.image}")
        print(f"  Gender     : {result['gender']}")
        print(f"  Confidence : {result['confidence']}%")
        print(f"  Inference  : {result['elapsed_ms']} ms")
        print(f"\n  Full probabilities:")
        for cls, prob in result["probabilities"].items():
            print(f"    {cls:<10} → {prob:>5.2f}%")
    except Exception as exc:
        print(f"[!] Error: {exc}")
        sys.exit(1)
