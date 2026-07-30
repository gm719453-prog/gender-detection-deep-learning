"""
Configuration file for the Gender Detection from Face project.
Contains all hyperparameters, paths, and settings in one central place.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Dataset paths ─────────────────────────────────────────────────────────────
DATASET_RAW_DIR       = PROJECT_ROOT / "dataset" / "raw"
DATASET_PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"

# UTKFace download URL
UTKFACE_DATASET_URL = "https://drive.google.com/uc?id=1GA2M4j8fJn8v3g3z5z5z5z5z5z5z5z5"

# ── Model settings ────────────────────────────────────────────────────────────
MODEL_SAVE_DIR       = PROJECT_ROOT / "models"
MODEL_FILE           = MODEL_SAVE_DIR / "gender_detection_model.keras"
MODEL_FILE_H5        = MODEL_SAVE_DIR / "gender_detection_model.h5"  # legacy

# Transfer learning backbone
BACKBONE             = "MobileNetV2"
INPUT_SHAPE          = (224, 224, 3)
NUM_CLASSES          = 2          # Male, Female
CLASS_NAMES          = ["Male", "Female"]

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE           = 64
EPOCHS               = 30
LEARNING_RATE        = 1e-3
LR_MIN               = 1e-6
LR_REDUCE_FACTOR     = 0.5
LR_REDUCE_PATIENCE   = 3

# ── Data augmentation ─────────────────────────────────────────────────────────
TRAIN_AUGMENTATION = {
    "rotation_range": 20,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.15,
    "horizontal_flip": True,
    "brightness_range": [0.8, 1.2],
    "fill_mode": "nearest",
}

# ── Train / val split ────────────────────────────────────────────────────────
VALIDATION_SPLIT     = 0.2
RANDOM_SEED          = 42

# ── Early stopping ────────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE  = 7
EARLY_STOP_MIN_DELTA = 0.001

# ── Preprocessing ─────────────────────────────────────────────────────────────
MEAN_SUBTRACTION     = True
NORMALIZATION        = True

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR              = PROJECT_ROOT / "logs"
