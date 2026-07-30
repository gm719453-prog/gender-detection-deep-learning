"""
preprocess.py – Dataset acquisition, parsing, and preprocessing.

This module handles:
  1. Downloading the UTKFace dataset (or pointing to a local copy).
  2. Parsing filenames to extract age, gender, and ethnicity labels.
  3. Building a pandas DataFrame with metadata.
  4. Splitting into train / validation sets.
  5. Loading and augmenting images via tf.data pipelines.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# ── Project config ────────────────────────────────────────────────────────────
from config import (
    PROJECT_ROOT, DATASET_RAW_DIR, DATASET_PROCESSED_DIR,
    INPUT_SHAPE, BATCH_SIZE, VALIDATION_SPLIT, RANDOM_SEED,
    TRAIN_AUGMENTATION,
)

# UTKFace filenames follow the pattern: [age]_[gender]_[race]_[date&time].jpg
# gender: 0 = Male, 1 = Female


# ── 1. Download the dataset ───────────────────────────────────────────────────
def download_dataset(
    dest_dir: Path = DATASET_RAW_DIR,
    kaggle_dataset: str = "prajnasb/observations",
) -> Path:
    """Download UTKFace (or a subset) into *dest_dir*.

    If the dataset is already present on disk the function returns immediately.
    We first try Kaggle, then fall back to Google Drive.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if faces are already extracted
    if list(dest_dir.glob("*.jpg")):
        print(f"[+] UTKFace images already present in {dest_dir}")
        return dest_dir

    # ── Attempt Kaggle download ───────────────────────────────────────────
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"[*] Downloading Kaggle dataset '{kaggle_dataset}' …")
        api.dataset_download_files(kaggle_dataset, path=str(dest_dir), unzip=True)
        # Kaggle download may place files in sub-folders; move them up
        _flatten(dest_dir)
        if list(dest_dir.glob("*.jpg")):
            print(f"[+] Download complete → {dest_dir}")
            return dest_dir
    except Exception as exc:
        print(f"[!] Kaggle download failed: {exc}")

    # ── Fallback: Google Drive via gdown ──────────────────────────────────
    try:
        import gdown
        url = "https://drive.google.com/uc?id=1GA2M4j8fJn8v3g3z5z5z5z5z5z5z5z5"
        zip_path = dest_dir / "UTKFace.zip"
        print("[*] Downloading from Google Drive …")
        gdown.download(url, str(zip_path), quiet=False)
        shutil.unpack_archive(str(zip_path), str(dest_dir))
        zip_path.unlink()
        _flatten(dest_dir)
        print(f"[+] Download complete → {dest_dir}")
        return dest_dir
    except Exception as exc:
        print(f"[!] Google Drive download failed: {exc}")

    # ── Last resort: provide instructions ─────────────────────────────────
    print("[!] Could not download dataset automatically.")
    print("[!] Please download UTKFace manually and place images in:")
    print(f"    {dest_dir}")
    raise RuntimeError("Dataset download failed – manual intervention required.")


def _flatten(directory: Path) -> None:
    """Move all files from subdirectories into *directory* (one level)."""
    for item in directory.iterdir():
        if item.is_dir():
            for f in item.iterdir():
                if f.is_file():
                    target = directory / f.name
                    if not target.exists():
                        shutil.move(str(f), str(target))
            # Remove empty dir
            try:
                item.rmdir()
            except OSError:
                pass


# ── 2. Parse filenames → DataFrame ────────────────────────────────────────────
def parse_dataset(raw_dir: Path = DATASET_RAW_DIR) -> pd.DataFrame:
    """Build a DataFrame with ``filepath``, ``age``, ``gender``, ``ethnicity``."""
    rows = []
    image_extensions = {".jpg", ".jpeg", ".png"}

    for file in sorted(raw_dir.iterdir()):
        if file.suffix.lower() not in image_extensions:
            continue
        parts = file.stem.split("_")
        if len(parts) < 2:
            continue  # skip files that don't match the naming convention

        try:
            age = int(parts[0])
            gender = int(parts[1])  # 0 = Male, 1 = Female
            ethnicity = int(parts[2]) if len(parts) > 2 else -1
            rows.append({
                "filepath": str(file.resolve()),
                "age": age,
                "gender": gender,
                "ethnicity": ethnicity,
            })
        except ValueError:
            continue  # skip malformed filenames

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid UTKFace images found. Check the dataset directory.")

    print(f"[+] Parsed {len(df)} images from {raw_dir}")
    print(df["gender"].value_counts().to_string())
    return df


# ── 3. Train / validation split ───────────────────────────────────────────────
def build_splits(
    df: pd.DataFrame,
    val_split: float = VALIDATION_SPLIT,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split by gender."""
    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=seed, stratify=df["gender"],
    )
    print(f"[+] Train samples: {len(train_df)} | Validation samples: {len(val_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ── 4. tf.data pipelines ──────────────────────────────────────────────────────
def _decode_and_preprocess(image_path: str, label, img_size: tuple[int, int] = (224, 224)):
    """Read a JPEG image and return (image, label) tensors."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.cast(image, tf.float32)
    )
    return image, label


def _augment(image, label):
    """Apply random augmentations during training."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # Random zoom (crop & resize back)
    if tf.random.uniform([]) > 0.5:
        zoom = tf.random.uniform([], 0.85, 1.0)
        h, w = tf.shape(image)[0], tf.shape(image)[1]
        new_h, new_w = tf.cast(tf.cast(h, tf.float32) * zoom, tf.int32), \
                       tf.cast(tf.cast(w, tf.float32) * zoom, tf.int32)
        image = tf.image.random_crop(image, size=[new_h, new_w, 3])
        image = tf.image.resize(image, (h, w))
    return image, label


def build_dataset(
    df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    img_size: tuple[int, int] = (224, 224),
    augment: bool = False,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Create a ``tf.data.Dataset`` from a DataFrame."""
    ds = tf.data.Dataset.from_tensor_slices(
        (df["filepath"].values, df["gender"].values)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, l: _decode_and_preprocess(p, l, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── 5. Convenience: full pipeline ─────────────────────────────────────────────
def prepare_all(
    raw_dir: Path = DATASET_RAW_DIR,
    processed_dir: Path = DATASET_PROCESSED_DIR,
) -> tuple[tf.data.Dataset, tf.data.Dataset, pd.DataFrame, pd.DataFrame]:
    """End-to-end: download → parse → split → build datasets."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = parse_dataset(raw_dir)

    # Save metadata
    df.to_csv(processed_dir / "full_metadata.csv", index=False)

    train_df, val_df = build_splits(df)
    train_df.to_csv(processed_dir / "train_metadata.csv", index=False)
    val_df.to_csv(processed_dir / "val_metadata.csv", index=False)

    train_ds = build_dataset(train_df, augment=True, shuffle=True)
    val_ds   = build_dataset(val_df,   augment=False, shuffle=False)

    print(f"[+] train_ds batches : {len(train_ds)}")
    print(f"[+] val_ds   batches : {len(val_ds)}")
    return train_ds, val_ds, train_df, val_df


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Gender Detection – Dataset Preprocessing")
    print("=" * 60)
    download_dataset()
    prepare_all()
    print("[✓] Preprocessing complete. You can now run train.py")
