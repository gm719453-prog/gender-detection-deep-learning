"""
app.py – Streamlit web application for Gender Detection from Face.

A premium, modern UI featuring:
  - Glassmorphism design with gradient background
  - Light / Dark mode toggle
  - Drag-and-drop image upload
  - Animated prediction card with confidence score
  - Loading animation during inference
  - Error handling for invalid images
  - Responsive, mobile-friendly layout
  - Professional typography and attractive icons

Optimized for Streamlit Community Cloud deployment.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# ── Suppress TensorFlow verbose logs on Streamlit Cloud ───────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_FILE, INPUT_SHAPE, CLASS_NAMES

# ── Streamlit page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gender Detection AI",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ── Custom CSS (Glassmorphism + Animations) ───────────────────────────────────
def inject_custom_css(theme: str = "light"):
    """Inject premium CSS styles into the Streamlit app."""

    if theme == "dark":
        bg_gradient = "linear-gradient(135deg, #0f0c29, #302b63, #24243e)"
        card_bg = "rgba(255, 255, 255, 0.08)"
        card_border = "rgba(255, 255, 255, 0.18)"
        text_primary = "#f0f0f0"
        text_secondary = "#b0b0b0"
        accent = "#7c4dff"
        accent_hover = "#651fff"
        shadow = "0 8px 32px rgba(0, 0, 0, 0.5)"
    else:
        bg_gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        card_bg = "rgba(255, 255, 255, 0.25)"
        card_border = "rgba(255, 255, 255, 0.4)"
        text_primary = "#2d2d2d"
        text_secondary = "#555555"
        accent = "#6c63ff"
        accent_hover = "#5a52d5"
        shadow = "0 8px 32px rgba(0, 0, 0, 0.15)"

    css = f"""
    <style>
        .stApp {{
            background: {bg_gradient};
            background-attachment: fixed;
            min-height: 100vh;
            font-family: 'Segoe UI', 'Inter', system-ui, -apple-system, sans-serif;
        }}
        .main .block-container {{
            background: transparent;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        h1, h2, h3 {{ color: {text_primary}; font-weight: 700; }}
        p, label, div {{ color: {text_secondary}; }}

        .glass-card {{
            background: {card_bg};
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid {card_border};
            border-radius: 20px;
            padding: 2rem;
            box-shadow: {shadow};
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .glass-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
        }}

        .result-card {{
            background: {card_bg};
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid {card_border};
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            box-shadow: {shadow};
            animation: fadeInUp 0.6s ease forwards;
        }}
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(30px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}

        .gender-badge {{
            display: inline-block;
            padding: 0.6rem 2rem;
            border-radius: 50px;
            font-size: 1.5rem;
            font-weight: 700;
            color: white;
            margin: 0.5rem 0;
        }}
        .gender-male   {{ background: linear-gradient(135deg, #4facfe, #00f2fe); }}
        .gender-female {{ background: linear-gradient(135deg, #f093fb, #f5576c); }}

        .confidence-bar {{
            width: 100%;
            height: 12px;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.15);
            margin: 1rem 0;
            overflow: hidden;
        }}
        .confidence-fill {{
            height: 100%;
            border-radius: 6px;
            background: linear-gradient(90deg, {accent}, {accent_hover});
            transition: width 0.8s ease;
        }}

        .spinner-container {{ text-align: center; padding: 2rem; }}
        .spinner {{
            border: 4px solid rgba(255, 255, 255, 0.15);
            border-top: 4px solid {accent};
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }}
        @keyframes spin {{
            0%   {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        .footer {{
            text-align: center;
            padding: 2rem 0 1rem;
            font-size: 0.85rem;
            color: {text_secondary};
        }}
        #MainMenu {{ visibility: hidden; }}
        header {{ visibility: hidden; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: str):
    """Load and cache the Keras model. Tries .keras then .h5 format."""
    # Try .keras format first (modern Keras), then fall back to .h5
    keras_path = model_path.replace(".h5", ".keras")
    if Path(keras_path).exists():
        return keras.models.load_model(keras_path)
    return keras.models.load_model(model_path)


# ── Prediction logic ──────────────────────────────────────────────────────────
def run_prediction(model, image_pil: Image.Image) -> dict:
    """Run inference on a PIL Image and return results."""
    img = image_pil.convert("RGB").resize((INPUT_SHAPE[0], INPUT_SHAPE[1]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    start = time.perf_counter()
    preds = model.predict(arr)
    elapsed = (time.perf_counter() - start) * 1000

    probs = preds[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx] * 100)

    return {
        "gender": CLASS_NAMES[idx],
        "confidence": round(confidence, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        },
        "elapsed_ms": round(elapsed, 2),
    }


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    # ── Theme toggle ──────────────────────────────────────────────────────
    col_theme1, col_theme2 = st.columns([6, 1])
    with col_theme2:
        theme = "dark" if st.checkbox("Dark", value=False) else "light"
    inject_custom_css(theme)

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.3rem;">Gender Detection AI</h1>
        <p style="font-size: 1.05rem; opacity: 0.85;">
            Powered by <strong>MobileNetV2</strong> Transfer Learning - Fast, Accurate & Beautiful
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Check model exists ────────────────────────────────────────────────
    model_path = str(MODEL_FILE)
    model_exists = Path(model_path).exists()

    # ── Upload section ────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Upload Face Image")
        st.markdown("<small>Accepts JPG, PNG, JPEG - Max 10 MB</small>", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear face image for gender detection.",
            key="uploader",
        )

        if uploaded is not None:
            try:
                image = Image.open(uploaded)
                image.verify()
                image = Image.open(uploaded)  # re-open after verify
            except (UnidentifiedImageError, OSError):
                st.error("Invalid or corrupted image file. Please upload a valid JPG, PNG, or JPEG image.")
                st.markdown('</div>', unsafe_allow_html=True)
                st.stop()

            # Show uploaded image
            st.markdown("---")
            st.markdown("### Uploaded Image")
            st.image(image, use_container_width=True)
            w, h = image.size
            st.caption(f"Dimensions: {w} x {h} px | Format: {uploaded.type or 'Unknown'}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 1rem; color: rgba(255,255,255,0.6);">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📷</div>
                <p style="font-size: 1.1rem;">Drag & drop a face image here,<br>or click to browse files.</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Prediction section ────────────────────────────────────────────────
    with col_right:
        if uploaded is not None:
            if not model_exists:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.error("""
                    **Model file not found!**
                    Please train the model first: `python train.py`
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                st.stop()

            model = load_keras_model(model_path)

            # Show spinner during prediction
            with st.spinner(""):
                st.markdown('<div class="spinner-container">', unsafe_allow_html=True)
                st.markdown('<div class="spinner"></div>', unsafe_allow_html=True)
                st.markdown('<p>Analyzing face...</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                result = run_prediction(model, image)

            # ── Result card ───────────────────────────────────────────────
            gender_lower = result["gender"].lower()
            badge_class = f"gender-{gender_lower}"

            st.markdown(f"""
            <div class="result-card">
                <h2 style="margin-top: 0; color: white;">Prediction Result</h2>
                <div class="gender-badge {badge_class}">
                    {result['gender']}
                </div>
                <p style="font-size: 1.1rem; margin-top: 1rem;">
                    <strong>Confidence:</strong> {result['confidence']:.1f}%
                </p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {result['confidence']}%;"></div>
                </div>
                <div style="margin-top: 1.2rem; font-size: 0.95rem;">
            """, unsafe_allow_html=True)

            for cls, prob in result["probabilities"].items():
                st.markdown(f"<p><strong>{cls}</strong> -> {prob}%</p>", unsafe_allow_html=True)

            st.markdown(f"""
                </div>
                <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
                    <small style="opacity: 0.7;">Inference time: <strong>{result['elapsed_ms']} ms</strong></small>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 4rem 2rem;">
                <div style="font-size: 5rem; margin-bottom: 1rem;">🎯</div>
                <h3 style="color: #f0f0f0;">Ready to Detect</h3>
                <p style="font-size: 1rem;">Upload a face image on the left to see<br>the AI prediction here.</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        <p>
            Built with <strong>TensorFlow + MobileNetV2</strong> |
            Trained on <strong>UTKFace Dataset</strong>
        </p>
        <p style="font-size: 0.75rem; opacity: 0.6;">
            MIT License - Gender Detection from Face
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
