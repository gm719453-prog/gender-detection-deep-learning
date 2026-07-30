"""
predict.py – Standalone inference for gender detection.

Compatible with Keras 3 (TensorFlow 2.16+).
"""

from __future__ import annotations

import argparse
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
    arr = np.expand_dims(arr, axis=0)
    return arr


# ── Model loading ─────────────────────────────────────────────────────────────
_model_cache = {}


def find_model_file() -> str:
    """Find the model file - try .keras first, then .h5."""
    model_dir = Path(MODEL_FILE).parent
    
    keras_path = str(model_dir / "gender_detection_model.keras")
    if Path(keras_path).exists():
        return keras_path
    
    h5_path = str(model_dir / "gender_detection_model.h5")
    if Path(h5_path).exists():
        return h5_path
    
    return str(MODEL_FILE)


def load_model(model_path: str | Path = None) -> keras.Model:
    """Load the trained Keras model (cached after first call)."""
    if model_path is None:
        model_path = find_model_file()
    else:
        model_path = str(model_path)
    
    if model_path not in _model_cache:
        print(f"[*] Loading model from {model_path} …")
        _model_cache[model_path] = keras.models.load_model(model_path)
    return _model_cache[model_path]


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_gender(
    image_path: str | Path,
    model_path: str | Path = None,
) -> dict:
    """Predict gender from a single face image."""
    model = load_model(model_path)

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    batch = preprocess_image(image_path)

    start = time.perf_counter()
    preds = model.predict(batch)
    elapsed = (time.perf_counter() - start) * 1000

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
    model_path: str | Path = None,
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
    parser.add_argument("--model", "-m", type=str, default=None, help="Path to the model file.")
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
