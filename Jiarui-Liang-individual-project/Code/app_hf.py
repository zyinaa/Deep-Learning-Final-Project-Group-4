# -*- coding: utf-8 -*-
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import os
import sys
from huggingface_hub import hf_hub_download
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing.dataset import (
    CLASS_INFO, IDX_TO_LABEL, NUM_CLASSES, compute_cancer_risk
)

st.set_page_config(
    page_title="DermAI",
    page_icon="icon.png",
    layout="wide"
)

def get_risk_level(score):
    if score < 20:
        return "Low Risk", "#00CC88", "✓"
    elif score < 50:
        return "Moderate Risk", "#FFD700", "⚠"
    elif score < 75:
        return "Elevated Risk", "#FF8C00", "⚠"
    else:
        return "High Risk", "#FF2D2D", "✗"

@st.cache_resource(show_spinner=False)
def load_model():
    model_path = hf_hub_download(
        repo_id="zyinaa/skin-cancer-vit",
        filename="efficientnet_b3_best.pth"
    )
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()
    return model, "cpu"

def preprocess_for_model(pil_image):
    img_arr = np.array(pil_image.convert("RGB"))
    h, w = img_arr.shape[:2]
    size = min(h, w)
    top  = (h - size) // 2
    left = (w - size) // 2
    img_arr = img_arr[top:top+size, left:left+size]
    lab = cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img_arr = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)
    display = cv2.resize(img_arr, (224, 224))
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tensor = transform(image=img_arr)["image"].unsqueeze(0)
    return tensor, display

def predict(model, tensor, device):
    with torch.no_grad():
        outputs = model(tensor.to(device))
        probs = torch.softmax(outputs, dim=-1)[0]
    return {IDX_TO_LABEL[i]: probs[i].item() for i in range(NUM_CLASSES)}

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px'>
        <h2 style='color:#00B4D8; margin:0'>DermAI</h2>
        <p style='color:#8899AA; font-size:12px'>Skin Cancer Risk System</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    **10 Classes:**
    Melanoma, BCC, SCC, Actinic Keratosis,
    Benign Keratosis, Nevi, Dermatofibroma,
    Vascular Lesion, Tinea, Healthy Skin

    **Photo Tips:**
    - Hold phone 5-10cm from skin
    - Use bright natural light
    - Keep lesion in focus
    """)
    st.divider()
    st.warning("For educational purposes only.")

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg, #0D1F3C, #1A3A5C);
     padding:20px 28px; border-radius:12px; margin-bottom:20px;
     border:1px solid #1E3A5F">
    <h1 style="color:white; margin:0; font-size:28px; font-weight:800">DermAI</h1>
    <p style="color:#8899AA; margin:4px 0 0 0; font-size:12px;
        letter-spacing:2px; text-transform:uppercase">
        Skin Cancer Risk Classification System
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load model with progress ──────────────────────────────
with st.spinner("Loading model..."):
    model, device = load_model()

st.markdown(
    "<div style='display:inline-flex; align-items:center; gap:8px;"
    "background:#0D1F3C; border:1px solid #1D9E75; border-radius:20px;"
    "padding:6px 14px; margin-bottom:16px'>"
    "<div style='width:8px; height:8px; background:#1D9E75;"
    "border-radius:50%'></div>"
    "<span style='color:#1D9E75; font-size:13px; font-weight:500'>"
    "Model Ready</span></div>",
    unsafe_allow_html=True
)

# ── Upload ────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg, #0D1F3C, #0A1A30);
     border-radius:12px; padding:20px 24px;
     border:1px solid #1E3A5F; margin-bottom:8px">
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px">
        <div style="width:3px; height:20px; background:#00B4D8; border-radius:2px"></div>
        <h3 style="color:white; margin:0; font-size:18px; font-weight:700">
            Upload Skin Image
        </h3>
    </div>
    <p style="color:#8899AA; margin:0; font-size:13px; padding-left:13px">
        Upload a close-up photo of the skin lesion
        &nbsp;•&nbsp; JPG, JPEG, PNG
    </p>
</div>
""", unsafe_allow_html=True)

img_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if img_file is not None:
    image = Image.open(img_file)

    col_img, col_btn = st.columns([1, 1])
    with col_img:
        st.image(image, caption="Uploaded image", width=200)
    with col_btn:
        st.write("")
        st.write("")
        st.write("")
        analyse_btn = st.button(
            "ANALYSE",
            type="primary",
            use_container_width=True
        )

    if analyse_btn:
        st.divider()
        progress = st.progress(0, text="Preprocessing...")
        tensor, display_img = preprocess_for_model(image)
        progress.progress(50, text="Running model...")
        probs_dict = predict(model, tensor, device)
        progress.progress(90, text="Computing risk score...")
        risk_score = compute_cancer_risk(probs_dict)
        top_cls    = max(probs_dict, key=probs_dict.get)
        confidence = probs_dict[top_cls]
        level, color, icon = get_risk_level(risk_score)
        progress.progress(100, text="Done!")
        progress.empty()

        # Results
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.metric("Cancer Risk Score", f"{risk_score} / 100")
        with col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")
        with col3:
            st.markdown(
                f"<div style='color:{color}; font-size:22px; font-weight:bold'>"
                f"{icon} {level}</div>"
                f"<p style='color:#8899AA; margin:4px 0'>Most likely: "
                f"<b style='color:white'>{CLASS_INFO[top_cls]['name']}</b></p>",
                unsafe_allow_html=True
            )

        if confidence < 0.30:
            st.warning("Low confidence — result may be unreliable.")

        st.divider()

        # Class probabilities
        st.subheader("Class Probabilities")
        HIGH_RISK   = ["mel", "scc", "bcc"]
        MEDIUM_RISK = ["akiec", "bkl", "nv"]
        for cls, prob in sorted(probs_dict.items(),
                               key=lambda x: x[1], reverse=True):
            if cls in HIGH_RISK:
                bar_color = "#FF2D2D"
            elif cls in MEDIUM_RISK:
                bar_color = "#FFD700"
            else:
                bar_color = "#00CC88"
            pct = prob * 100
            st.markdown(
                f"<div style='display:flex; align-items:center; margin:4px 0'>"
                f"<div style='width:200px; color:white; font-size:14px'>"
                f"{CLASS_INFO[cls]['name']}</div>"
                f"<div style='flex:1; background:#1A2F4A; border-radius:4px;"
                f"height:18px; margin:0 10px'>"
                f"<div style='width:{pct:.1f}%; background:{bar_color};"
                f"border-radius:4px; height:18px'></div></div>"
                f"<div style='width:50px; color:white; font-size:14px;"
                f"text-align:right'>{pct:.1f}%</div></div>",
                unsafe_allow_html=True
            )

        st.divider()
        st.info(
            "Educational use only. Not a medical diagnostic tool. "
            "Always consult a qualified dermatologist."
        )
