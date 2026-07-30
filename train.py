"""
train.py – Train the Gender Detection model using MobileNetV2 transfer learning.

Features implemented:
  • MobileNetV2 base (pre-trained on ImageNet, frozen initially)
  • Custom classification head (GlobalAveragePooling2D → Dense → Dropout → Dense)
  • Adam optimizer with cosine-annealed learning rate
  • Data augmentation applied via tf.data pipeline
  • Early stopping (monitoring val_loss)
  • Model checkpoint (save best model by val_accuracy)
  • Learning rate reduction on plateau
  • Training history visualisation
  • Evaluation: accuracy, precision, recall, F1 score, classification report
  • Confusion matrix visualisation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks

# Project imports
from config import (
    MODEL_SAVE_DIR, MODEL_FILE, INPUT_SHAPE, NUM_CLASSES, CLASS_NAMES,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, LR_MIN, LR_REDUCE_FACTOR,
    LR_REDUCE_PATIENCE, EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA,
    LOG_DIR, RANDOM_SEED, VALIDATION_SPLIT,
)
from preprocess import prepare_all
from utils.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_sample_predictions,
)
from utils.helpers import get_device_info

# Reproducibility
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── 1. Model architecture ─────────────────────────────────────────────────────
def build_model(
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    trainable_base: bool = False,
) -> keras.Model:
    """Construct a MobileNetV2 transfer-learning model."""

    # Base model – pre-trained on ImageNet, exclude top
    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    if not trainable_base:
        base.trainable = False
    else:
        # Fine-tune: unfreeze layers from index 100 onwards
        for layer in base.layers[:100]:
            layer.trainable = False
        for layer in base.layers[100:]:
            layer.trainable = True

    # Custom head
    x = base.output
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, name="dropout1")(x)
    x = layers.Dense(128, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=base.input, outputs=outputs, name="GenderClassifier")
    return model


# ── 2. Learning-rate scheduler ────────────────────────────────────────────────
def get_callbacks(model_path: Path = MODEL_FILE) -> list[keras.callbacks.Callback]:
    """Return a list of Keras callbacks for training."""
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Early stopping
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA,
        restore_best_weights=True,
        verbose=1,
    )

    # Model checkpoint
    checkpoint = callbacks.ModelCheckpoint(
        str(model_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    # Reduce LR on plateau
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE,
        min_lr=LR_MIN,
        verbose=1,
    )

    # TensorBoard
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tb = callbacks.TensorBoard(log_dir=str(LOG_DIR), histogram_freq=1)

    return [early_stop, checkpoint, reduce_lr, tb]


# ── 3. Training function ──────────────────────────────────────────────────────
def train(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    initial_lr: float = LEARNING_RATE,
) -> keras.callbacks.History:
    """Full training pipeline: two-phase (frozen base → fine-tune)."""

    print("\n" + "=" * 60)
    print("  Gender Detection – Model Training")
    print("=" * 60)
    print(get_device_info())

    # ── Prepare data ──────────────────────────────────────────────────────
    print("\n[*] Preparing dataset …")
    train_ds, val_ds, train_df, val_df = prepare_all()

    steps_per_epoch  = len(train_ds)
    validation_steps = len(val_ds)

    # ── Phase 1: Frozen base ──────────────────────────────────────────────
    print("\n[*] Phase 1 – Training with frozen base …")
    model = build_model(trainable_base=False)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=initial_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    cb_phase1 = get_callbacks()

    history_phase1 = model.fit(
        train_ds,
        epochs=10,  # Phase 1 runs for 10 epochs max
        validation_data=val_ds,
        callbacks=cb_phase1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
    )

    # ── Phase 2: Fine-tune ────────────────────────────────────────────────
    print("\n[*] Phase 2 – Fine-tuning top layers …")
    model_finetune = build_model(trainable_base=True)
    # Load best weights from phase 1
    if MODEL_FILE.exists():
        model_finetune.load_weights(str(MODEL_FILE))
        print("[+] Loaded best weights from Phase 1")

    model_finetune.compile(
        optimizer=optimizers.Adam(learning_rate=initial_lr / 10),  # lower LR
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    cb_phase2 = get_callbacks()

    history_phase2 = model_finetune.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=cb_phase2,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
    )

    # ── Merge histories ───────────────────────────────────────────────────
    merged = {
        k: list(history_phase1.history[k]) + list(history_phase2.history[k])
        for k in history_phase1.history
    }

    # ── Save model ────────────────────────────────────────────────────────
    model_finetune.save(str(MODEL_FILE))
    print(f"\n[✓] Model saved to {MODEL_FILE}")

    # ── Save training history as JSON ─────────────────────────────────────
    history_path = MODEL_SAVE_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in merged.items()}, f, indent=2)
    print(f"[✓] Training history saved to {history_path}")

    return merged


# ── 4. Evaluation ─────────────────────────────────────────────────────────────
def evaluate(model_path: Path = MODEL_FILE, raw_dir=None):
    """Load the best model and compute classification metrics."""
    from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

    print("\n" + "=" * 60)
    print("  Gender Detection – Model Evaluation")
    print("=" * 60)

    model = keras.models.load_model(str(model_path))
    print(f"[+] Loaded model from {model_path}")

    # Use validation split
    from preprocess import parse_dataset, build_splits, build_dataset
    if raw_dir is None:
        raw_dir = __import__("config").DATASET_RAW_DIR
    df = parse_dataset(raw_dir)
    _, val_df = build_splits(df)
    val_ds = build_dataset(val_df, augment=False, shuffle=False)

    # Predict
    y_true = []
    y_pred = []
    sample_images = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        if len(sample_images) < 12:
            for img, lbl, pred in zip(images.numpy(), labels.numpy(), np.argmax(preds, axis=1)):
                if len(sample_images) < 12:
                    sample_images.append((img, int(lbl), int(pred)))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Metrics
    accuracy  = float(np.mean(y_true == y_pred))
    precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall    = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1        = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    print(f"\n{'Metric':<15} {'Value':>8}")
    print("-" * 25)
    print(f"{'Accuracy':<15} {accuracy:>8.4f}")
    print(f"{'Precision':<15} {precision:>8.4f}")
    print(f"{'Recall':<15} {recall:>8.4f}")
    print(f"{'F1 Score':<15} {f1:>8.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    # Visualisations
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, save_path=MODEL_SAVE_DIR / "confusion_matrix.png")

    if sample_images:
        imgs = [s[0] for s in sample_images]
        trues = [s[1] for s in sample_images]
        preds = [s[2] for s in sample_images]
        plot_sample_predictions(imgs, trues, preds, CLASS_NAMES, save_path=MODEL_SAVE_DIR / "sample_predictions.png")

    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
    metrics_path = MODEL_SAVE_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[✓] Metrics saved to {metrics_path}")

    return metrics


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Train
    history = train()

    # Visualise training curves
    plot_training_history(history, save_path=MODEL_SAVE_DIR / "training_curves.png")

    # Evaluate
    evaluate()

    print("\n" + "=" * 60)
    print("  ✓ Training & Evaluation Complete!")
    print("=" * 60)
