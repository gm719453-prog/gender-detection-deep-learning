# Gender Detection from Face

A production-quality deep learning project that detects gender (Male / Female) from a face image using **MobileNetV2** transfer learning. The project includes a complete training pipeline, a standalone prediction module, and a **premium Streamlit web application** with a glassmorphism UI.

---

## Table of Contents

1. [Features](#features)
2. [Technology Stack](#technology-stack)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Dataset Setup](#dataset-setup)
6. [Training](#training)
7. [Prediction (CLI)](#prediction-cli)
8. [Running the Web Application](#running-the-web-application)
9. [Model Architecture](#model-architecture)
10. [Performance Metrics](#performance-metrics)
11. [Screenshots](#screenshots)
12. [Future Enhancements](#future-enhancements)
13. [Troubleshooting](#troubleshooting)
14. [Contributing](#contributing)
15. [License](#license)

---

## Features

- **Accurate Gender Detection** – Trained on the UTKFace dataset (~20 000+ face images) with transfer learning.
- **MobileNetV2 Backbone** – Lightweight, fast inference suitable for edge devices.
- **Premium Web UI** – Glassmorphism design, gradient backgrounds, animated buttons, and responsive layout.
- **Light & Dark Mode** – Toggle between themes with a single click.
- **Drag-and-Drop Upload** – Accepts JPG, PNG, and JPEG files.
- **Confidence Score** – Displays prediction probability with an animated progress bar.
- **Fast Inference** – Sub-100 ms prediction on CPU.
- **Error Handling** – Validates uploaded images and shows helpful error messages.
- **Loading Animation** – Spinner displayed during model inference.
- **Mobile-Friendly** – Responsive design that works on phones and tablets.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning Framework | TensorFlow 2.15+ / Keras |
| Model Architecture | MobileNetV2 (Transfer Learning) |
| Image Processing | OpenCV, Pillow |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Web Application | Streamlit |
| Dataset | UTKFace |

---

## Project Structure

```
Gender-Detection-DeepLearning/
├── app.py                  # Streamlit web application
├── train.py                # Training pipeline
├── predict.py              # Standalone prediction module
├── preprocess.py           # Dataset download & preprocessing
├── config.py               # Central configuration file
├── requirements.txt        # Python dependencies
├── README.md               # This documentation
├── LICENSE                 # MIT License
├── dataset/
│   ├── raw/                # Downloaded UTKFace images
│   └── processed/          # Metadata CSVs
├── models/
│   ├── gender_detection_model.h5   # Trained model weights
│   ├── training_history.json       # Training metrics
│   ├── confusion_matrix.png        # Evaluation plot
│   ├── sample_predictions.png      # Sample visualisation
│   └── training_curves.png         # Loss & accuracy curves
├── notebooks/
│   └── exploration.ipynb   # Jupyter notebook for EDA
├── utils/
│   ├── __init__.py
│   ├── visualization.py    # Plotting utilities
│   └── helpers.py          # General helpers
└── logs/                   # TensorBoard logs
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Gender-Detection-DeepLearning.git
cd Gender-Detection-DeepLearning
```

### 2. Create a Virtual Environment (Recommended)

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Kaggle API for Dataset Download

If you want to download the dataset from Kaggle:

```bash
pip install kaggle
# Follow instructions at https://www.kaggle.com/docs/api to set up your API token
mkdir ~/.kaggle
# Place kaggle.json in ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Dataset Setup

### Automatic Download

Run the preprocessing script to download the UTKFace dataset automatically:

```bash
python preprocess.py
```

The script will:
1. Attempt to download from Kaggle (if API is configured).
2. Fall back to Google Drive.
3. Parse filenames to extract gender labels (0 = Male, 1 = Female).
4. Create train / validation splits (80 / 20 stratified).
5. Build `tf.data` pipelines with augmentation.

### Manual Download

If automatic download fails:

1. Download UTKFace from [Kaggle](https://www.kaggle.com/datasets/prajnasb/observations) or [Google Drive](https://drive.google.com/drive/folders/0B4UoNlZnJW5uY1YxQ2dGZ0dGZ0E).
2. Extract the zip file and place all `.jpg` images into `dataset/raw/`.
3. Run `python preprocess.py` to build the data pipelines.

### UTKFace Dataset Details

| Property | Value |
|----------|-------|
| Total images | ~23,708 |
| Classes | Male (0), Female (1) |
| Image size | Variable (resized to 224×224) |
| Filename format | `[age]_[gender]_[race]_[date].jpg` |
| Train / Val split | 80% / 20% (stratified) |

---

## Training

### Run the Full Training Pipeline

```bash
python train.py
```

The training script performs a **two-phase** approach:

| Phase | Description | Epochs | Learning Rate |
|-------|-------------|--------|---------------|
| Phase 1 | Train classification head with frozen base | Up to 10 | 1e-3 |
| Phase 2 | Fine-tune top layers of MobileNetV2 | Up to 30 | 1e-4 |

### Training Features

- **Adam Optimizer** with configurable learning rate.
- **Early Stopping** (patience = 7, monitors `val_loss`).
- **Model Checkpoint** (saves best model by `val_accuracy`).
- **Learning Rate Reduction** on plateau (factor = 0.5, patience = 3).
- **Data Augmentation** (random flip, brightness, contrast, zoom).
- **TensorBoard** logging for real-time monitoring.

### TensorBoard

```bash
tensorboard --logdir logs/
```

Open `http://localhost:6006` in your browser.

---

## Prediction (CLI)

Predict gender from a single face image:

```bash
python predict.py --image path/to/face.jpg
```

Or with a custom model path:

```bash
python predict.py --model models/gender_detection_model.h5 --image face.jpg
```

**Sample output:**

```
==================================================
  Gender Detection – Prediction
==================================================

  Image      : path/to/face.jpg
  Gender     : Male
  Confidence : 94.23%
  Inference  : 45.12 ms

  Full probabilities:
    Male       → 94.23%
    Female     →  5.77%
```

---

## Running the Web Application

### Start the Streamlit App

```bash
streamlit run app.py
```

### Access the Application

Open your browser and navigate to:

```
http://localhost:8501
```

### Web App Features

- **Upload face image** via drag-and-drop or file browser.
- **Instant prediction** with confidence score and animated progress bar.
- **Light / Dark mode** toggle in the top-right corner.
- **Glassmorphism UI** with gradient background.
- **Responsive design** that works on mobile devices.

---

## Model Architecture

The model uses **MobileNetV2** as a feature extractor with a custom classification head:

```
Input: 224 × 224 × 3 (RGB)
    │
    ▼
MobileNetV2 (pre-trained, ImageNet weights)
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
Dense(256, ReLU) + Dropout(0.5)
    │
    ▼
Dense(128, ReLU) + Dropout(0.3)
    │
    ▼
Dense(2, Softmax)
    │
    ▼
Output: [Male_prob, Female_prob]
```

### Why MobileNetV2?

- **Lightweight** – Only 3.4 million parameters (base model).
- **Fast inference** – Optimized for mobile and edge devices.
- **Strong features** – Pre-trained on ImageNet with inverted residual blocks.
- **Efficient** – Depthwise separable convolutions reduce computation.

---

## Performance Metrics

Typical results after training on UTKFace:

| Metric | Value |
|--------|-------|
| Accuracy | ~92 – 96% |
| Precision (weighted) | ~92 – 96% |
| Recall (weighted) | ~92 – 96% |
| F1 Score (weighted) | ~92 – 96% |
| Inference Time | ~30 – 80 ms |

> **Note:** Exact metrics depend on the dataset version, augmentation settings, and number of training epochs.

---

## Screenshots

### Web Application

| Light Mode | Dark Mode |
|------------|-----------|
| Glassmorphism upload card with gradient background | Dark glassmorphism with animated prediction |

### Training Visualisations

| Training Curves | Confusion Matrix |
|-----------------|------------------|
| Loss & accuracy over epochs | True vs. predicted labels |

### Sample Predictions

| Correct Prediction | Confidence Bar |
|--------------------|----------------|
| Green label for correct, red for incorrect | Animated fill showing confidence |

> *Add your own screenshots by running the app and capturing the UI.*

---

## Future Enhancements

- **Age estimation** – Extend the model to predict age in addition to gender.
- **Ethnicity classification** – Add a multi-class head for race/ethnicity prediction.
- **Real-time webcam inference** – Stream webcam frames and predict in real time.
- **REST API** – Expose the model as a FastAPI / Flask endpoint.
- **Model quantization** – Convert to TFLite for mobile deployment.
- **Batch processing** – Upload multiple images and get batch predictions.
- **Face detection pre-processing** – Auto-crop faces using MTCNN or RetinaFace before classification.
- **Multi-language support** – Add i18n to the web interface.
- **User accounts** – Store prediction history with user authentication.
- **A/B testing** – Compare MobileNetV2 with EfficientNet, ResNet, or Vision Transformers.

---

## Troubleshooting

### Model File Not Found

```bash
python train.py
```

### CUDA / GPU Issues

If you encounter TensorFlow GPU errors:

```bash
# Force CPU-only execution
export CUDA_VISIBLE_DEVICES=""
```

Or install the CPU-only version:

```bash
pip install tensorflow-cpu
```

### Memory Errors

Reduce the batch size in `config.py`:

```python
BATCH_SIZE = 32  # or even 16
```

### Streamlit Port Already in Use

```bash
streamlit run app.py --server.port 8502
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this project in your research or portfolio, please cite:

```bibtex
@misc{gender-detection-face-2024,
  author = {Your Name},
  title = {Gender Detection from Face using MobileNetV2},
  year = {2024},
  howpublished = {\url{https://github.com/your-username/Gender-Detection-DeepLearning}},
}
```

---

**Built with ❤️ using TensorFlow & Streamlit**
