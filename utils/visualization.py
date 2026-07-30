"""
Visualization utilities for training results and predictions.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers / notebooks

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ── Plot training history ─────────────────────────────────────────────────────
def plot_training_history(history: dict, save_path: str | Path = "training_history.png"):
    """Plot training and validation loss & accuracy from Keras history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["loss"]) + 1)

    # Loss plot
    axes[0].plot(epochs, history["loss"], "b-", label="Training Loss")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Validation Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history["accuracy"], "b-", label="Training Accuracy")
    axes[1].plot(epochs, history["val_accuracy"], "r-", label="Validation Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Training history plot saved to {save_path}")


# ── Plot confusion matrix ─────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str | Path = "confusion_matrix.png",
):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")

    # Annotate with percentages
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                ha="center", va="center", fontsize=11,
            )

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Confusion matrix saved to {save_path}")


# ── Plot sample predictions ───────────────────────────────────────────────────
def plot_sample_predictions(
    images: list[np.ndarray],
    true_labels: list[int],
    pred_labels: list[int],
    class_names: list[str],
    save_path: str | Path = "sample_predictions.png",
    num_samples: int = 12,
):
    """Display a grid of sample images with true and predicted labels."""
    n = min(num_samples, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(n):
        row, col = idx // cols, idx % cols
        axes[row][col].imshow(images[idx])
        color = "green" if true_labels[idx] == pred_labels[idx] else "red"
        title = f"True: {class_names[true_labels[idx]]}\nPred: {class_names[pred_labels[idx]]}"
        axes[row][col].set_title(title, color=color, fontsize=10)
        axes[row][col].axis("off")

    # Hide unused subplots
    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Sample predictions saved to {save_path}")
