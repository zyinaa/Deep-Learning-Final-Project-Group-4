# -*- coding: utf-8 -*-
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
import matplotlib.cm as cm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.dataset import (
    CLASS_INFO, LABEL_TO_IDX, IDX_TO_LABEL,
    NUM_CLASSES, compute_cancer_risk, preprocess_image
)

from transformers import ViTForImageClassification
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Skin Cancer Risk Analyser",
    page_icon="🔬",
    layout="wide"
)

# ── Paths ─────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models/saved/vit_best.pth")

# ── Risk Level ────────────────────────────────────────────
def get_risk_level(score):
    if score < 20:
        return "Low Risk", "#1D9E75", "✓"
    elif score < 50:
        return "Moderate Risk", "#EF9F27", "⚠"
    elif score < 75:
        return "Elevated Risk", "#D85A30", "⚠"
    else:
        return "High Risk", "#E24B4A", "✗"

# ── Load Model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
        output_attentions=True,
    )
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(
            torch.load(MODEL_PATH,
                      map_location="cpu")
        )
        st.sidebar.success("Model loaded!")
    else:
        st.sidebar.error(f"Model not found: {MODEL_PATH}")

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    return model, device

# ── Preprocessing ─────────────────────────────────────────
def preprocess_for_model(pil_image):
    img_arr = np.array(pil_image.convert("RGB"))

    # Center crop to square
    h, w = img_arr.shape[:2]
    size = min(h, w)
    top  = (h - size) // 2
    left = (w - size) // 2
    img_arr = img_arr[top:top+size, left:left+size]

    # CLAHE
    lab = cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img_arr = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)

    # Resize for display
    display = cv2.resize(img_arr, (224, 224))

    # Transform for model
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
    tensor = transform(image=img_arr)["image"].unsqueeze(0)

    return tensor, display

# ── Attention Map ─────────────────────────────────────────
def get_attention_map(model, tensor, device):
    with torch.no_grad():
        outputs = model(
            tensor.to(device),
            output_attentions=True,
            output_hidden_states=True
        )
        # Last layer attention, average over heads
        # Shape: [1, 12, 197, 197]
        attn = outputs.attentions[-1]
        # CLS token attention to all patches
        attn_map = attn[0].mean(0)[0, 1:]  # [196]
        attn_map = attn_map.reshape(14, 14).cpu().numpy()

    # Normalize
    attn_map = (attn_map - attn_map.min())
    attn_map = attn_map / (attn_map.max() + 1e-8)

    # Resize to 224x224
    attn_map = cv2.resize(attn_map, (224, 224))
    return attn_map

# ── Overlay Attention ─────────────────────────────────────
def overlay_attention(img_arr, attn_map):
    heatmap = cm.jet(attn_map)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img_arr, 0.55, heatmap, 0.45, 0)
    return overlay

# ── Predict ───────────────────────────────────────────────
def predict(model, tensor, device):
    with torch.no_grad():
        outputs = model(
            tensor.to(device),
            output_attentions=True,
            output_hidden_states=True
        )
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    probs_dict = {
        IDX_TO_LABEL[i]: probs[i].item()
        for i in range(NUM_CLASSES)
    }
    return probs_dict, outputs

# ════════════════════════════════════════════════════════
#  MAIN UI
# ════════════════════════════════════════════════════════

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 About")
    st.markdown("""
    **Skin Cancer Risk Analyser**

    Powered by Vision Transformer (ViT)
    trained on HAM10000 + supplementary data.

    **10 Classes:**
    - Melanoma, BCC, SCC, Actinic Keratosis
    - Benign Keratosis, Nevi, Dermatofibroma
    - Vascular Lesion, Tinea, Healthy Skin

    **How to use:**
    1. Upload or capture a skin image
    2. Get instant risk assessment
    3. View attention map explanation
    """)
    st.divider()
    st.markdown("""
    **Photo Tips:**
    - Hold phone 5-10cm from skin
    - Use bright natural light
    - Keep lesion in focus
    - Fill frame with the lesion
    """)
    st.divider()
    st.warning(
        "For educational purposes only. "
        "Not a medical diagnostic tool."
    )

# ── Main Title ────────────────────────────────────────────
st.title("🔬 Skin Cancer Risk Analyser")
st.caption(
    "Deep Learning Final Project — Group 4 | "
    "GWU DATS 6303 | ViT-base-patch16-224"
)
st.divider()

# ── Load Model ────────────────────────────────────────────
model, device = load_model()

# ── Input Tabs ────────────────────────────────────────────
tab1, tab2 = st.tabs(["📷 Live Capture", "📁 Upload Image"])

img_file = None
with tab1:
    img_file = st.camera_input(
        "Point camera at skin lesion and capture"
    )

with tab2:
    uploaded = st.file_uploader(
        "Upload a skin image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        img_file = uploaded

# ── Run Inference ─────────────────────────────────────────
if img_file is not None:
    image = Image.open(img_file)

    # Quality check
    img_check = np.array(image.convert("RGB"))
    gray      = cv2.cvtColor(img_check, cv2.COLOR_RGB2GRAY)
    blur      = cv2.Laplacian(gray, cv2.CV_64F).var()

    # blur check disabled

    with st.spinner("Analysing..."):
        tensor, display_img = preprocess_for_model(image)
        probs_dict, outputs  = predict(model, tensor, device)
        attn_map             = get_attention_map(
            model, tensor, device
        )
        overlay              = overlay_attention(
            display_img, attn_map
        )
        risk_score           = compute_cancer_risk(probs_dict)
        top_cls              = max(probs_dict, key=probs_dict.get)
        confidence           = probs_dict[top_cls]
        level, color, icon   = get_risk_level(risk_score)

    st.divider()

    # ── Score Display ─────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.metric(
            label="Cancer Risk Score",
            value=f"{risk_score} / 100"
        )

    with col2:
        st.metric(
            label="Confidence",
            value=f"{confidence*100:.1f}%"
        )

    with col3:
        st.markdown(
            f"<h3 style='color:{color}'>"
            f"{icon} {level}</h3>",
            unsafe_allow_html=True
        )
        st.write(
            f"Most likely: **{CLASS_INFO[top_cls]['name']}** "
            f"({probs_dict[top_cls]*100:.1f}%)"
        )

    if confidence < 0.40:
        st.warning(
            "Low confidence — image may not resemble "
            "conditions in our training data."
        )

    st.divider()

    # ── Images ────────────────────────────────────────────
    st.subheader("What the model focused on")
    img_col, attn_col = st.columns(2)
    img_col.image(
        display_img,
        caption="Preprocessed image",
        use_column_width=True
    )
    attn_col.image(
        overlay,
        caption="Attention map",
        use_column_width=True
    )

    st.divider()

    # ── Probability Bars ──────────────────────────────────
    st.subheader("Class Probabilities")

    sorted_probs = sorted(
        probs_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for cls, prob in sorted_probs:
        col_name, col_bar, col_pct = st.columns([3, 5, 1])
        col_name.write(CLASS_INFO[cls]["name"])
        col_bar.progress(float(prob))
        col_pct.write(f"{prob*100:.1f}%")

    st.divider()

    # ── Disclaimer ────────────────────────────────────────
    st.info(
        "**Educational use only.** "
        "This tool is not a medical device and cannot "
        "diagnose skin cancer. Always consult a qualified "
        "dermatologist for any skin concern."
    )
