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
import matplotlib.cm as cm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import ViTForImageClassification

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.dataset import (
    CLASS_INFO, IDX_TO_LABEL, NUM_CLASSES, compute_cancer_risk
)

st.set_page_config(
    page_title="Skin Cancer Risk Analyser",
    page_icon="🔬",
    layout="wide"
)

# Load icon as base64
import base64 as _b64
_icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
with open(_icon_path, "rb") as _f:
    _icon_b64 = _b64.b64encode(_f.read()).decode()

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIT_PATH   = os.path.join(BASE, "models/saved/vit_best.pth")
ENET_PATH  = os.path.join(BASE, "models/saved/efficientnet_b3_best.pth")

def get_risk_level(score):
    if score < 20:
        return "Low Risk", "#00CC88", "✓"
    elif score < 50:
        return "Moderate Risk", "#FFD700", "⚠"
    elif score < 75:
        return "Elevated Risk", "#FF8C00", "⚠"
    else:
        return "High Risk", "#FF2D2D", "✗"

def create_gauge(score, level, color):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0A1628")
    ax.set_facecolor("#0A1628")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, 1.3)
    ax.axis("off")
    ax.set_aspect("equal")

    # Draw colored zones
    zones = [
        (0,   20,  "#1D9E75"),
        (20,  50,  "#EF9F27"),
        (50,  75,  "#D85A30"),
        (75,  100, "#E24B4A"),
    ]

    for z_min, z_max, z_color in zones:
        theta1 = 180 - (z_min / 100) * 180
        theta2 = 180 - (z_max / 100) * 180
        wedge = patches.Wedge(
            (0, 0), 1.1,
            theta2, theta1,
            width=0.25,
            facecolor=z_color,
            alpha=0.85,
            edgecolor="#0A1628",
            linewidth=1.5
        )
        ax.add_patch(wedge)

    # Draw score arc overlay (brighter)
    score_theta1 = 180
    score_theta2 = 180 - (score / 100) * 180
    score_wedge = patches.Wedge(
        (0, 0), 1.1,
        score_theta2, score_theta1,
        width=0.25,
        facecolor=color,
        alpha=0.3,
        edgecolor="none"
    )
    ax.add_patch(score_wedge)

    # Needle
    angle_rad = np.radians(180 - (score / 100) * 180)
    needle_x = 0.85 * np.cos(angle_rad)
    needle_y = 0.85 * np.sin(angle_rad)
    ax.annotate("",
        xy=(needle_x, needle_y),
        xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="-|>",
            color="white",
            lw=2,
            mutation_scale=12
        )
    )

    # Center circle
    center = patches.Circle((0, 0), 0.08,
                            facecolor="white",
                            edgecolor="#0A1628",
                            linewidth=2, zorder=5)
    ax.add_patch(center)

    # Score text
    ax.text(0, -0.2, f"{score}", ha="center", va="center",
           fontsize=38, fontweight="bold", color="white")
    ax.text(0, -0.35, "/ 100", ha="center", va="center",
           fontsize=13, color="#8899AA")
    ax.text(0, -0.47, level, ha="center", va="center",
           fontsize=14, fontweight="bold", color=color)

    # Labels
    ax.text(-1.2, -0.05, "0", ha="center", va="center",
           fontsize=13, color="#8899AA")
    ax.text(1.2, -0.05, "100", ha="center", va="center",
           fontsize=13, color="#8899AA")
    ax.text(0, 1.2, "Cancer Risk", ha="center", va="center",
           fontsize=12, color="#8899AA")

    plt.tight_layout(pad=0.5)
    return fig


def section_header(title, color="#00B4D8"):
    st.markdown(
        f"<div style='display:flex; align-items:center; gap:12px; "
        f"margin:24px 0 16px 0'>"
        f"<div style='width:4px; height:28px; background:{color}; "
        f"border-radius:2px; flex-shrink:0'></div>"
        f"<span style='color:white; font-size:20px; font-weight:700'>"
        f"{title}</span></div>",
        unsafe_allow_html=True
    )


def section_header(title, color="#00B4D8"):
    st.markdown(
        f"<div style='display:flex; align-items:center; gap:12px; "
        f"margin:24px 0 16px 0'>"
        f"<div style='width:4px; height:28px; background:{color}; "
        f"border-radius:2px; flex-shrink:0'></div>"
        f"<span style='color:white; font-size:20px; font-weight:700'>"
        f"{title}</span></div>",
        unsafe_allow_html=True
    )


@st.cache_resource
def load_vit():
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
        output_attentions=True,
    )
    model.load_state_dict(torch.load(VIT_PATH, map_location="cpu"))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    dummy = torch.zeros(1, 3, 224, 224).to(device)
    with torch.no_grad():
        model(dummy, output_attentions=True)
    return model, device

@st.cache_resource
def load_efficientnet():
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(ENET_PATH, map_location="cpu"))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, device

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
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
    tensor = transform(image=img_arr)["image"].unsqueeze(0)
    return tensor, display

def get_attention_map(model, tensor, device):
    with torch.no_grad():
        outputs = model(
            tensor.to(device),
            output_attentions=True,
            output_hidden_states=True
        )
        attn = outputs.attentions[-1]
        attn_map = attn[0].mean(0)[0, 1:].reshape(14, 14).cpu().numpy()
    attn_map = (attn_map - attn_map.min())
    attn_map = attn_map / (attn_map.max() + 1e-8)
    attn_map = cv2.resize(attn_map, (224, 224))
    return attn_map

def overlay_attention(img_arr, attn_map):
    heatmap = cm.jet(attn_map)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    return cv2.addWeighted(img_arr, 0.55, heatmap, 0.45, 0)

def predict_vit(model, tensor, device):
    with torch.no_grad():
        outputs = model(
            tensor.to(device),
            output_attentions=True,
            output_hidden_states=True
        )
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    return {IDX_TO_LABEL[i]: probs[i].item() for i in range(NUM_CLASSES)}

def predict_efficientnet(model, tensor, device):
    with torch.no_grad():
        outputs = model(tensor.to(device))
        probs = torch.softmax(outputs, dim=-1)[0]
    return {IDX_TO_LABEL[i]: probs[i].item() for i in range(NUM_CLASSES)}

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")

    # Apply dark blue theme permanently
    st.markdown("""
    <style>
    .stApp {
        background-color: #0A1628;
        color: #FAFAFA;
    }
    .stSidebar {
        background-color: #0D1F3C;
    }
    h1, h2 {
        color: #00B4D8 !important;
    }
    .stMetric {
        background-color: #0D1F3C;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton > button {
        background: linear-gradient(180deg, #00C4E8 0%, #0096C7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(180deg, #00D4F8 0%, #00A6D7 100%);
        
        transform: translateY(-1px);
    }
    .stButton > button:active {
        transform: translateY(1px);
        
    }
    .stButton > button[kind="primary"] {
        background: #0A1628 !important;
        border: 2px solid #00B4D8 !important;
        color: #00B4D8 !important;
        font-size: 18px !important;
        font-weight: 900 !important;
        letter-spacing: 4px !important;
        text-transform: uppercase !important;
        padding: 14px 28px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        font-family: "Inter", "Helvetica Neue", Arial, sans-serif !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #00B4D8 !important;
        color: white !important;
        border-color: #00B4D8 !important;
    }
    .stDownloadButton > button {
        background: linear-gradient(180deg, #1DB87A 0%, #158A5A 100%);
        color: white;
        border: none;
        border-radius: 8px;
        
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(180deg, #22CC88 0%, #1A9E68 100%);
        
        transform: translateY(-1px);
    }
    .stProgress > div > div > div > div {
        background-color: #1D9E75 !important;
    }
    [data-testid="stProgressBar"] {
        background-color: #1D9E75 !important;
    }
    div[role="progressbar"] > div {
        background-color: #1D9E75 !important;
    }
    div[data-testid="stProgress"] > div {
        background-color: #00B4D8;
    }
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #0D1F3C, #0A1A30);
        border: 1.5px solid #1E3A5F;
        border-radius: 12px;
        padding: 20px;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #00B4D8;
        background: linear-gradient(135deg, #0A2040, #0D2545);
    }
    [data-testid="stFileUploader"] label p {
        color: white !important;
        font-size: 21px !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif !important;
    }
    [data-testid="stFileUploader"] label {
        color: #00B4D8 !important;
        font-size: 15px !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #8899AA !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #00B4D8 !important;
        font-weight: 600 !important;
    }
    </style>
    <script>
    function addPulse() {
        const btns = window.parent.document.querySelectorAll('button[kind="primary"]');
        btns.forEach(btn => {
            btn.style.cssText += `
                background: #00B4D8 !important;
                font-weight: 900 !important;
                font-size: 18px !important;
                letter-spacing: 3px !important;
                position: relative !important;
                overflow: hidden !important;
            `;
        });
    }
    setTimeout(addPulse, 1000);
    setInterval(addPulse, 2000);
    </script>
    """, unsafe_allow_html=True)

    # Model Selection
    st.markdown("""
    <div style="background:#0A1628; border-radius:10px; padding:14px;
         border:1px solid #1E3A5F; margin-bottom:12px">
        <p style="color:#00B4D8; font-size:16px; font-weight:700;
            letter-spacing:2px; text-transform:uppercase; margin:0 0 10px 0">
            Model Selection
        </p>
    </div>
    """, unsafe_allow_html=True)

    model_choice = st.radio(
        "Choose model:",
        [
            "ViT-base (Main Model)",
            "EfficientNet-B3 (Baseline)",
            "Both (Compare)"
        ],
        index=0,
        label_visibility="collapsed"
    )

    # Reset results when model changes
    if "last_model_choice" not in st.session_state:
        st.session_state.last_model_choice = model_choice
    if st.session_state.last_model_choice != model_choice:
        st.session_state.results = None
        st.session_state.last_model_choice = model_choice

    confidence_threshold = 0.30

    st.divider()

    # Demo Mode
    st.markdown("""
    <div style="margin-bottom:8px">
        <p style="color:#00B4D8; font-size:14px; font-weight:700;
            letter-spacing:2px; text-transform:uppercase; margin:0">
            Demo Mode
        </p>
        <p style="color:#8899AA; font-size:12px; margin:4px 0 0 0">
            Use pre-selected HAM10000 test images
        </p>
    </div>
    """, unsafe_allow_html=True)

    demo_mode = st.toggle("Use demo images", value=False)

    if demo_mode:
        demo_images = {
            "Melanoma (High Risk)":        "data/raw/ham10000/HAM10000_images_part_1/ISIC_0025964.jpg",
            "Melanocytic Nevi (Low Risk)":  "data/raw/ham10000/HAM10000_images_part_1/ISIC_0024306.jpg",
            "Basal Cell Carcinoma":         "data/raw/ham10000/HAM10000_images_part_1/ISIC_0024310.jpg",
            "Benign Keratosis":             "data/raw/ham10000/HAM10000_images_part_1/ISIC_0024307.jpg",
            "Healthy Skin":                 "data/raw/kaggle_diseases/Oily-Dry-Skin-Types/train/normal/1 (1).jpg",
        }
        selected_demo = st.selectbox(
            "Select a demo case:",
            list(demo_images.keys()),
            label_visibility="collapsed"
        )

        # Reset when demo image changes
        if st.session_state.get("last_demo") != selected_demo:
            st.session_state.results = None
            st.session_state.analysis_done = False
            st.session_state.last_demo = selected_demo

    st.divider()

    # Model Info
    if "model_choice" not in dir() or model_choice == "ViT-base (Main Model)":
        arch = "ViT-base-patch16"
        f1   = "0.7652"
        acc  = "86%"
        attn = "Self-Attention"
    elif model_choice == "EfficientNet-B3 (Baseline)":
        arch = "EfficientNet-B3"
        f1   = "0.7440"
        acc  = "83%"
        attn = "N/A"
    else:
        arch = "ViT + EfficientNet"
        f1   = "0.7652 / 0.7440"
        acc  = "86% / 83%"
        attn = "Self-Attention"

    st.markdown(f"""
    <div style="background:#0A1628; border-radius:10px; padding:14px;
         border:1px solid #1E3A5F; margin-bottom:12px">
        <p style="color:#00B4D8; font-size:16px; font-weight:700;
            letter-spacing:2px; text-transform:uppercase; margin:0 0 10px 0">
            Model Info
        </p>
        <div style="display:flex; justify-content:space-between; margin-bottom:8px">
            <span style='color:#8899AA; font-size:15px'>Architecture</span>
            <span style='color:white; font-size:15px; font-weight:600'>{arch}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:8px">
            <span style='color:#8899AA; font-size:15px'>Test F1</span>
            <span style='color:#00CC88; font-size:15px; font-weight:600'>{f1}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:8px">
            <span style='color:#8899AA; font-size:15px'>Accuracy</span>
            <span style='color:#00CC88; font-size:15px; font-weight:600'>{acc}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:8px">
            <span style='color:#8899AA; font-size:15px'>Explainability</span>
            <span style='color:white; font-size:15px; font-weight:600'>{attn}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:8px">
            <span style='color:#8899AA; font-size:15px'>Classes</span>
            <span style='color:white; font-size:15px; font-weight:600'>10</span>
        </div>
        <div style="display:flex; justify-content:space-between">
            <span style='color:#8899AA; font-size:15px'>Training Images</span>
            <span style='color:white; font-size:15px; font-weight:600'>13,215</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <p style="color:#4A5568; font-size:11px; text-align:center">
        For educational purposes only.<br/>
        Not a medical diagnostic tool.
    </p>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────
# Header banner
st.markdown(f"""
<div style="background: linear-gradient(135deg, #0D1F3C 0%, #1A3A5C 50%, #0D1F3C 100%);
     padding: 24px 32px; border-radius: 12px; margin-bottom: 24px;
     border: 1px solid #1E3A5F;
     display: flex; align-items: center; justify-content: space-between">
    <div>
        <div style="display: flex; align-items: center; gap: 14px">
            <img src="data:image/png;base64,{_icon_b64}" style="width:108px; height:108px; border-radius:12px"/>
            <div>
                <span style="color: #00B4D8; font-size: 34px; font-weight: 900;
                    letter-spacing: -1px; font-family: Georgia, serif">Derm</span><span
                    style="color: white; font-size: 34px; font-weight: 900;
                    letter-spacing: -1px; font-family: Georgia, serif">AI</span>
                <div style="width: 100%; height: 2px;
                    background: linear-gradient(90deg, #00B4D8, transparent);
                    margin-top: 2px"></div>
            </div>
        </div>
        <p style="color: #8899AA; margin: 6px 0 0 0; font-size: 13px;
            font-weight: 400; letter-spacing: 2px; text-transform: uppercase;">
            Skin Cancer Risk Classification System
        </p>
    </div>
    <div style="text-align: right">
        <p style="color: #8899AA; margin: 0; font-size: 12px">
            Powered by ViT-base-patch16-224
        </p>
        <p style="color: #8899AA; margin: 4px 0 0 0; font-size: 12px">
            GWU DATS 6303 | Group 4
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# Load models based on selection
if model_choice == "ViT-base (Main Model)":
    with st.spinner("Loading ViT model..."):
        vit_model, device = load_vit()
    st.markdown(
    "<div style='display:inline-flex; align-items:center; gap:8px; "
    "background:#0D1F3C; border:1px solid #1D9E75; border-radius:20px; "
    "padding:6px 14px; margin-bottom:8px'>"
    "<div style='width:8px; height:8px; background:#1D9E75; "
    "border-radius:50%'></div>"
    "<span style='color:#1D9E75; font-size:13px; font-weight:500'>"
    "Model Ready</span></div>",
    unsafe_allow_html=True
)
elif model_choice == "EfficientNet-B3 (Baseline)":
    with st.spinner("Loading EfficientNet model..."):
        enet_model, device = load_efficientnet()
    st.markdown(
    "<div style='display:inline-flex; align-items:center; gap:8px; "
    "background:#0D1F3C; border:1px solid #1D9E75; border-radius:20px; "
    "padding:6px 14px; margin-bottom:8px'>"
    "<div style='width:8px; height:8px; background:#1D9E75; "
    "border-radius:50%'></div>"
    "<span style='color:#1D9E75; font-size:13px; font-weight:500'>"
    "Model Ready</span></div>",
    unsafe_allow_html=True
)
else:
    with st.spinner("Loading both models..."):
        vit_model, device  = load_vit()
        enet_model, device = load_efficientnet()
    st.markdown(
    "<div style='display:inline-flex; align-items:center; gap:8px; "
    "background:#0D1F3C; border:1px solid #1D9E75; border-radius:20px; "
    "padding:6px 14px; margin-bottom:8px'>"
    "<div style='width:8px; height:8px; background:#1D9E75; "
    "border-radius:50%'></div>"
    "<span style='color:#1D9E75; font-size:13px; font-weight:500'>"
    "Both Models Ready</span></div>",
    unsafe_allow_html=True
)

# Upload or Demo
image = None

if "demo_mode" in dir() and demo_mode:
    st.info(f"Demo mode: **{selected_demo}**")
    demo_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        demo_images[selected_demo]
    )
    if os.path.exists(demo_path):
        image = Image.open(demo_path)
else:
    img_file = st.file_uploader(
        "Upload your Dermoscopy Photo",
        type=["jpg", "jpeg", "png"],
    )
    if img_file is not None:
        # Reset results when new image uploaded
        if "last_img" not in st.session_state or st.session_state.last_img != img_file.name:
            st.session_state.results = None
            st.session_state.analysis_done = False
            st.session_state.last_img = img_file.name
        image = Image.open(img_file)



if image is not None:

    col_img, col_btn = st.columns([1, 1])
    with col_img:
        st.image(image, caption="Uploaded image", width=200)
    with col_btn:
        st.write("")
        st.write("")
        st.write("")
        analyse_btn = st.button(
            "SCAN",
            type="primary",
            use_container_width=True
        )

    if analyse_btn:
        st.divider()

        progress = st.progress(0, text="Preprocessing...")
        tensor, display_img = preprocess_for_model(image)
        progress.progress(30, text="Running model(s)...")

        if model_choice == "ViT-base (Main Model)":
            probs_dict = predict_vit(vit_model, tensor, device)
            progress.progress(70, text="Generating attention map...")
            attn_map = get_attention_map(vit_model, tensor, device)
            overlay  = overlay_attention(display_img, attn_map)
            show_attention = True
            model_label = "ViT-base"

        elif model_choice == "EfficientNet-B3 (Baseline)":
            probs_dict = predict_efficientnet(enet_model, tensor, device)
            show_attention = False
            model_label = "EfficientNet-B3"
            risk_score_e = compute_cancer_risk(probs_dict)
            top_cls_e    = max(probs_dict, key=probs_dict.get)
            level_e, color_e, icon_e = get_risk_level(risk_score_e)
            st.session_state.results = {
                "probs_dict":    probs_dict,
                "risk_score":    risk_score_e,
                "top_cls":       top_cls_e,
                "confidence":    probs_dict[top_cls_e],
                "level":         level_e,
                "color":         color_e,
                "icon":          icon_e,
                "model_label":   "EfficientNet-B3",
                "show_attention": False,
                "is_comparison": False,
            }
            st.session_state.display_img = display_img
            overlay = None
            st.session_state.overlay = None

        else:
            vit_probs  = predict_vit(vit_model, tensor, device)
            enet_probs = predict_efficientnet(enet_model, tensor, device)
            progress.progress(70, text="Generating attention map...")
            attn_map = get_attention_map(vit_model, tensor, device)
            overlay  = overlay_attention(display_img, attn_map)
            show_attention = True
            model_label = "Comparison"

            # Save comparison to session state
            vit_risk_comp  = compute_cancer_risk(vit_probs)
            enet_risk_comp = compute_cancer_risk(enet_probs)
            st.session_state.results = {
                "probs_dict":    vit_probs,
                "risk_score":    vit_risk_comp,
                "top_cls":       max(vit_probs, key=vit_probs.get),
                "confidence":    vit_probs[max(vit_probs, key=vit_probs.get)],
                "level":         get_risk_level(vit_risk_comp)[0],
                "color":         get_risk_level(vit_risk_comp)[1],
                "icon":          get_risk_level(vit_risk_comp)[2],
                "model_label":   "Comparison",
                "show_attention": True,
                "enet_probs":    enet_probs,
                "enet_risk":     enet_risk_comp,
                "is_comparison": True,
            }
            st.session_state.display_img = display_img
            st.session_state.overlay = overlay

        progress.progress(100, text="Done!")
        progress.empty()

        st.markdown(
            "<div style='display:inline-flex; align-items:center; gap:8px;"
            "background:#0D1F3C; border:1px solid #00CC88; border-radius:20px;"
            "padding:8px 18px; margin-bottom:16px'>"
            "<div style='width:10px; height:10px; background:#00CC88;"
            "border-radius:50%'></div>"
            "<span style='color:#00CC88; font-size:14px; font-weight:600;"
            "letter-spacing:1px'>ANALYSIS COMPLETE</span></div>",
            unsafe_allow_html=True
        )
        st.session_state.analysis_done = True

        # Clinical Recommendations
        RECOMMENDATIONS = {
            "low": {
                "color": "#00CC88",
                "icon": "✓",
                "title": "Low Risk — Routine Monitoring",
                "actions": [
                    "Continue monthly self-examination",
                    "Annual full-body skin check with a dermatologist",
                    "Use SPF 30+ sunscreen daily",
                    "Document any changes in size, shape, or color",
                ]
            },
            "moderate": {
                "color": "#FFD700",
                "icon": "⚠",
                "title": "Moderate Risk — Schedule a Check-up",
                "actions": [
                    "Schedule a dermatologist appointment within 1–3 months",
                    "Photograph the lesion to track changes over time",
                    "Avoid UV exposure and use SPF 50+ sunscreen",
                    "Do not attempt to remove or treat the lesion yourself",
                ]
            },
            "high": {
                "color": "#FF2D2D",
                "icon": "✗",
                "title": "High Risk — Seek Immediate Consultation",
                "actions": [
                    "Consult a dermatologist as soon as possible",
                    "Consider requesting a dermoscopy evaluation",
                    "A biopsy may be required for definitive diagnosis",
                    "Do not delay — early detection significantly improves outcomes",
                ]
            }
        }




        # ── Single model results ──────────────────────────
        if model_choice != "Both (Compare)":
            risk_score = compute_cancer_risk(probs_dict)
            top_cls    = max(probs_dict, key=probs_dict.get)
            confidence = probs_dict[top_cls]
            level, color, icon = get_risk_level(risk_score)

            # Show clinical recommendation
            if risk_score < 20:
                rec = RECOMMENDATIONS["low"]
            elif risk_score < 50:
                rec = RECOMMENDATIONS["moderate"]
            else:
                rec = RECOMMENDATIONS["high"]

            actions_html = "".join([
                f"<div style='display:flex; align-items:flex-start; gap:10px; margin:8px 0'>"
                f"<div style='width:6px; height:6px; background:{rec['color']}; "
                f"border-radius:50%; margin-top:6px; flex-shrink:0'></div>"
                f"<span style='color:#CCDDEE; font-size:17px'>{a}</span></div>"
                for a in rec["actions"]
            ])

            st.markdown(
                f"<div style='background:#0D1F3C; border-radius:12px; padding:20px;"
                f"border:1px solid {rec['color']}44; margin-bottom:16px'>"
                f"<div style='color:{rec['color']}; font-size:19px; font-weight:700;"
                f"margin-bottom:12px'>{rec['icon']} {rec['title']}</div>"
                f"{actions_html}"
                f"<p style='color:#4A5568; font-size:11px; margin-top:12px; margin-bottom:0'>"
                f"AI-generated guidance only. Always consult a qualified dermatologist.</p>"
                f"</div>",
                unsafe_allow_html=True
            )


            # Save to session state
            st.session_state.results = {
                "probs_dict": probs_dict,
                "risk_score": risk_score,
                "top_cls": top_cls,
                "confidence": confidence,
                "level": level,
                "color": color,
                "icon": icon,
                "model_label": model_label,
                "show_attention": show_attention,
            }
            st.session_state.display_img = display_img
            st.session_state.overlay = overlay

            section_header(f"Results — {model_label}")

            col_gauge, col_info = st.columns([1, 2])
            with col_gauge:
                import matplotlib.pyplot as plt
                gauge_fig = create_gauge(risk_score, level, color)
                st.pyplot(gauge_fig, use_container_width=True)
                plt.close()

            with col_info:
                st.markdown(
                    f"<div style='background:#0D1F3C; border-radius:12px;"
                    f"padding:20px; margin-top:10px'>"
                    f"<div style='color:{color}; font-size:24px; font-weight:bold; margin:0 0 12px 0'>{icon} {level}</div>"
                    f"<p style='color:#8899AA; margin:8px 0 4px'>Most likely condition</p>"
                    f"<h4 style='color:white; margin:0'>{CLASS_INFO[top_cls]['name']}</h4>"
                    f"<p style='color:#8899AA; margin:8px 0 4px'>Confidence</p>"
                    f"<h4 style='color:white; margin:0'>{confidence*100:.1f}%</h4>"
                    f"<p style='color:#8899AA; margin:8px 0 4px'>Model</p>"
                    f"<h4 style='color:white; margin:0'>{model_label}</h4>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            if confidence < confidence_threshold:
                st.warning(
                    f"Confidence {confidence*100:.1f}% is below "
                    f"your threshold {confidence_threshold*100:.0f}%. "
                    "Result may be unreliable."
                )


            if confidence < confidence_threshold:
                st.warning(
                    f"Confidence {confidence*100:.1f}% is below threshold. "
                    "Result may be unreliable."
                )




            if show_attention:
                st.divider()
                section_header("What the Model Focused On")
                st.markdown(
                    "<p style='color:#8899AA; font-size:13px; margin-bottom:16px'>"
                    "The attention map shows which regions drove the model prediction — "
                    "analogous to the ABCDE criteria used by dermatologists. "
                    "Red/Yellow = High attention | Blue/Purple = Low attention</p>",
                    unsafe_allow_html=True
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.image(display_img, width=400)
                    st.markdown(
                        "<p style='color:#8899AA; font-size:12px; text-align:center;"
                        "text-transform:uppercase; letter-spacing:1px'>"
                        "Preprocessed Image</p>",
                        unsafe_allow_html=True
                    )
                with c2:
                    st.image(overlay, width=400)
                    st.markdown(
                        "<p style='color:#00B4D8; font-size:12px; text-align:center;"
                        "text-transform:uppercase; letter-spacing:1px'>"
                        "Attention Map</p>",
                        unsafe_allow_html=True
                    )

            st.divider()

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors as rl_colors
            from reportlab.lib.units import cm
            import io

            HIGH_RISK   = ["mel", "scc", "bcc"]
            MEDIUM_RISK = ["akiec", "bkl", "nv"]
            LOW_RISK    = ["df", "vasc", "tinea", "normal"]

            high_prob   = sum(probs_dict.get(c, 0) for c in HIGH_RISK)
            medium_prob = sum(probs_dict.get(c, 0) for c in MEDIUM_RISK)
            low_prob    = sum(probs_dict.get(c, 0) for c in LOW_RISK)

            groups = [
                ("High Risk",   HIGH_RISK,   ["#E24B4A","#C0392B","#922B21"],   "#E24B4A", high_prob),
                ("Medium Risk", MEDIUM_RISK, ["#EF9F27","#F0B429","#F5C842"],   "#EF9F27", medium_prob),
                ("Low Risk",    LOW_RISK,    ["#1D9E75","#27AE60","#2ECC71","#58D68D"], "#1D9E75", low_prob),
            ]

            section_header("Class Probabilities")
            HIGH_RISK_C   = ["mel", "scc", "bcc"]
            MEDIUM_RISK_C = ["akiec", "bkl", "nv"]
            for cls, prob in sorted(
                probs_dict.items(), key=lambda x: x[1], reverse=True
            ):
                if cls in HIGH_RISK_C:
                    bar_color = "#E24B4A"
                elif cls in MEDIUM_RISK_C:
                    bar_color = "#EF9F27"
                else:
                    bar_color = "#1D9E75"
                pct = prob * 100
                st.markdown(
                    f"<div style='display:flex; align-items:center; margin:4px 0'>"
                    f"<div style='width:180px; color:white; font-size:13px'>"
                    f"{CLASS_INFO[cls]['name']}</div>"
                    f"<div style='flex:1; background:#1A2F4A; border-radius:4px; height:16px; margin:0 10px'>"
                    f"<div style='width:{pct:.1f}%; background:{bar_color}; "
                    f"border-radius:4px; height:16px'></div></div>"
                    f"<div style='width:45px; color:white; font-size:13px; text-align:right'>"
                    f"{pct:.1f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.divider()
            section_header("Risk Group Distribution")

            for group_name, classes, colors, title_color, group_prob in groups:
                import io, base64

                # Generate donut chart
                fig, ax = plt.subplots(figsize=(5, 5))
                fig.patch.set_alpha(0)
                fig.patch.set_facecolor("none")
                ax.set_facecolor("none")

                vals = [max(probs_dict.get(c, 0), 0.001) for c in classes]

                # 3D Donut chart
                # Shadow layer
                shadow_result = ax.pie(
                    vals,
                    colors=["#000000"] * len(vals),
                    startangle=88,
                    radius=1.02,
                    wedgeprops={"edgecolor": "none", "linewidth": 0,
                               "width": 0.58, "alpha": 0.3}
                )

                # Main donut - no labels inside
                wedges, texts = ax.pie(
                    vals,
                    colors=colors,
                    startangle=90,
                    wedgeprops={"edgecolor": "none", "linewidth": 0,
                               "width": 0.55, "antialiased": True}
                )


                # Highlight edge (lighter inner ring)
                highlight_result = ax.pie(
                    vals,
                    colors=["white"] * len(vals),
                    startangle=90,
                    radius=0.46,
                    wedgeprops={"edgecolor": "none", "linewidth": 0,
                               "width": 0.04, "alpha": 0.15}
                )

                # Center circle for depth
                center_circle = plt.Circle((0, 0), 0.42,
                                          fc="#0A1628", zorder=10)
                ax.add_patch(center_circle)

                # Center text
                ax.text(0, 0.08, f"{group_prob*100:.0f}%",
                       ha="center", va="center",
                       fontsize=22, fontweight="bold", color="white",
                       zorder=11)
                ax.text(0, -0.18, "of total",
                       ha="center", va="center",
                       fontsize=10, color="#8899AA", zorder=11)

                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format="png", transparent=True,
                           bbox_inches="tight", dpi=120)
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode()
                plt.close()

                # Card with donut + list
                col_pie, col_list = st.columns([1, 1])

                with col_pie:
                    st.markdown(
                        f"<div style='background:#0D1F3C; border-radius:12px;"
                        f"border:1px solid {title_color}44; padding:16px;"
                        f"text-align:center; margin-bottom:16px'>"
                        f"<div style='color:{title_color}; font-size:16px;"
                        f"font-weight:700; margin-bottom:12px; letter-spacing:1px'>"
                        f"{group_name}</div>"
                        f"<img src='data:image/png;base64,{img_b64}' "
                        f"style='width:280px; height:280px; display:block; margin:auto'/>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                with col_list:
                    list_items = ""
                    for cls, color in zip(classes, colors):
                        prob = probs_dict.get(cls, 0)
                        pct = prob * 100
                        list_items += (
                            f"<div style='margin:10px 0'>"
                            f"<div style='display:flex; align-items:center;"
                            f"justify-content:space-between; margin-bottom:4px'>"
                            f"<div style='display:flex; align-items:center; gap:8px'>"
                            f"<div style='width:12px; height:12px; background:{color};"
                            f"border-radius:3px; flex-shrink:0'></div>"
                            f"<span style='color:white; font-size:14px; font-weight:600'>"
                            f"{CLASS_INFO[cls]['name']}</span></div>"
                            f"<span style='color:{color}; font-size:14px; font-weight:700'>"
                            f"{pct:.1f}%</span></div>"
                            f"<div style='background:#1A2F4A; border-radius:4px; height:6px'>"
                            f"<div style='width:{min(prob/max(group_prob,0.001)*100, 100):.1f}%; background:{color};"
                            f"border-radius:4px; height:6px'></div></div>"
                            f"</div>"
                        )
                    st.markdown(
                        f"<div style='background:#0D1F3C; border-radius:12px;"
                        f"border:1px solid #1E3A5F; padding:16px; margin-bottom:16px'>"
                        f"{list_items}</div>",
                        unsafe_allow_html=True
                    )



        # ── Comparison mode ───────────────────────────────
        else:
            vit_risk  = compute_cancer_risk(vit_probs)
            enet_risk = compute_cancer_risk(enet_probs)
            vit_top   = max(vit_probs,  key=vit_probs.get)
            enet_top  = max(enet_probs, key=enet_probs.get)
            vit_level,  vit_color,  vit_icon  = get_risk_level(vit_risk)
            enet_level, enet_color, enet_icon = get_risk_level(enet_risk)

            section_header("Model Comparison")
            col_vit, col_enet = st.columns(2)

            with col_vit:
                st.markdown("### ViT-base (Main Model)")
                st.metric("Cancer Risk Score", f"{vit_risk} / 100")
                st.markdown(
                    f"<h4 style='color:{vit_color}'>{vit_icon} {vit_level}</h4>",
                    unsafe_allow_html=True
                )
                st.write(
                    f"Most likely: **{CLASS_INFO[vit_top]['name']}** "
                    f"({vit_probs[vit_top]*100:.1f}%)"
                )

            with col_enet:
                st.markdown("### EfficientNet-B3 (Baseline)")
                st.metric("Cancer Risk Score", f"{enet_risk} / 100")
                st.markdown(
                    f"<h4 style='color:{enet_color}'>{enet_icon} {enet_level}</h4>",
                    unsafe_allow_html=True
                )
                st.write(
                    f"Most likely: **{CLASS_INFO[enet_top]['name']}** "
                    f"({enet_probs[enet_top]*100:.1f}%)"
                )

            st.divider()
            section_header("Attention Map (ViT)")
            c1, c2 = st.columns(2)
            c1.image(display_img, caption="Preprocessed", width=250)
            c2.image(overlay, caption="ViT Attention map", width=250)

            st.divider()
            section_header("Risk Group Distribution — ViT vs EfficientNet")

            HIGH_RISK   = ["mel", "scc", "bcc"]
            MEDIUM_RISK = ["akiec", "bkl", "nv"]
            LOW_RISK    = ["df", "vasc", "tinea", "normal"]

            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(14, 9))
            fig.patch.set_facecolor("#0E1117")

            for row, (pdict, model_name) in enumerate([
                (vit_probs, "ViT-base"),
                (enet_probs, "EfficientNet-B3")
            ]):
                high_vals   = [pdict.get(c, 0) for c in HIGH_RISK]
                med_vals    = [pdict.get(c, 0) for c in MEDIUM_RISK]
                low_vals    = [pdict.get(c, 0) for c in LOW_RISK]
                high_labels = [CLASS_INFO[c]["name"] for c in HIGH_RISK]
                med_labels  = [CLASS_INFO[c]["name"] for c in MEDIUM_RISK]
                low_labels  = [CLASS_INFO[c]["name"] for c in LOW_RISK]

                high_prob = sum(high_vals)
                med_prob  = sum(med_vals)
                low_prob  = sum(low_vals)

                for col, (vals, labels, colors, title, color) in enumerate([
                    (high_vals, high_labels, ["#E24B4A","#D85A30","#C0392B"],
                     f"High Risk\n{high_prob*100:.1f}%", "#E24B4A"),
                    (med_vals,  med_labels,  ["#EF9F27","#F0B429","#F5C842"],
                     f"Medium Risk\n{med_prob*100:.1f}%", "#EF9F27"),
                    (low_vals,  low_labels,  ["#1D9E75","#27AE60","#2ECC71","#58D68D"],
                     f"Low Risk\n{low_prob*100:.1f}%", "#1D9E75"),
                ]):
                    axes[row][col].pie(
                        vals, labels=labels, colors=colors,
                        autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
                        startangle=90,
                        textprops={"color": "white", "fontsize": 8},
                        wedgeprops={"edgecolor": "#0E1117", "linewidth": 2}
                    )
                    axes[row][col].set_title(
                        f"{model_name}\n{title}",
                        color=color, fontsize=10, fontweight="bold"
                    )
                    axes[row][col].set_facecolor("#0E1117")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.divider()
        st.info(
            "Educational use only. Not a medical diagnostic tool. "
            "Always consult a qualified dermatologist."
        )



# ── PDF Report (outside analyse block) ───────────────────
# ── PDF Report (outside analyse block) ───────────────────
if (st.session_state.get("results") is not None
    and st.session_state.results.get("model_label") != "Comparison"
    and image is not None
    and st.session_state.get("analysis_done", False)):

    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm

    if st.button("Generate PDF Report", type="secondary", key="pdf_btn_main"):
        r           = st.session_state.results
        probs_dict  = r["probs_dict"]
        risk_score  = r["risk_score"]
        top_cls     = r["top_cls"]
        confidence  = r["confidence"]
        model_label = r["model_label"]
        is_comparison = r.get("is_comparison", False)
        enet_probs  = r.get("enet_probs", None)
        enet_risk   = r.get("enet_risk", None)
        level_text, _, _ = get_risk_level(risk_score)

        HIGH_RISK   = ["mel", "scc", "bcc"]
        MEDIUM_RISK = ["akiec", "bkl", "nv"]
        LOW_RISK    = ["df", "vasc", "tinea", "normal"]
        high_prob   = sum(probs_dict.get(c, 0) for c in HIGH_RISK)
        medium_prob = sum(probs_dict.get(c, 0) for c in MEDIUM_RISK)
        low_prob    = sum(probs_dict.get(c, 0) for c in LOW_RISK)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import tempfile, os
        from reportlab.platypus import PageBreak, Image as RLImage

        HIGH_RISK_C   = ["mel", "scc", "bcc"]
        MEDIUM_RISK_C = ["akiec", "bkl", "nv"]
        LOW_RISK_C    = ["df", "vasc", "tinea", "normal"]

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        h1_style = ParagraphStyle("h1", fontSize=16, spaceAfter=16,
                                  fontName="Helvetica-Bold",
                                  textColor=rl_colors.HexColor("#1D3A5F"))
        h2_style = ParagraphStyle("h2", fontSize=12, spaceAfter=6,
                                  spaceBefore=12, fontName="Helvetica-Bold",
                                  textColor=rl_colors.HexColor("#1D3A5F"))
        normal = ParagraphStyle("n", fontSize=10, fontName="Helvetica",
                               textColor=rl_colors.HexColor("#2C2C2A"))

        # ── PAGE 1 ────────────────────────────────────────
        # Title
        story.append(Paragraph("Skin Cancer Risk Assessment Report", h1_style))


        # Risk score bar
        fig_bar, ax_bar = plt.subplots(figsize=(10, 1.2))
        fig_bar.patch.set_facecolor("white")
        ax_bar.set_facecolor("white")
        # Background bar
        ax_bar.barh(0, 100, height=0.5, color="#F0F0F0", edgecolor="none")
        # Score bar
        score_color = "#E24B4A" if risk_score >= 75 else                      "#D85A30" if risk_score >= 50 else                      "#EF9F27" if risk_score >= 20 else "#1D9E75"
        ax_bar.barh(0, risk_score, height=0.5,
                   color=score_color, edgecolor="none")
        ax_bar.set_xlim(0, 100)
        ax_bar.set_yticks([])
        ax_bar.set_xlabel("Cancer Risk Score (0-100)", fontsize=10)
        ax_bar.text(risk_score + 1, 0, f"{risk_score}/100",
                   va="center", fontsize=12, fontweight="bold",
                   color=score_color)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["left"].set_visible(False)
        plt.tight_layout()
        tmp_scorebar = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        plt.savefig(tmp_scorebar.name, dpi=150, bbox_inches="tight",
                   facecolor="white")
        plt.close()
        story.append(RLImage(tmp_scorebar.name, width=16*cm, height=2*cm))
        story.append(Spacer(1, 0.3*cm))

        # Risk summary table
        story.append(Paragraph("Risk Summary", h2_style))
        risk_data = [
            ["Cancer Risk Score", f"{risk_score} / 100"],
            ["Risk Level",        level_text],
            ["Most Likely",       CLASS_INFO[top_cls]["name"]],
            ["Confidence",        f"{confidence*100:.1f}%"],
            ["Model",             model_label],
        ]
        rt = Table(risk_data, colWidths=[8*cm, 8*cm])
        rt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (0,-1), rl_colors.HexColor("#E6F1FB")),
            ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME",      (1,0), (1,-1), "Helvetica"),
            ("FONTSIZE",      (0,0), (-1,-1), 10),
            ("GRID",          (0,0), (-1,-1), 0.5, rl_colors.HexColor("#CCCCCC")),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("ROWBACKGROUNDS",(0,0), (-1,-1),
             [rl_colors.HexColor("#E6F1FB"), rl_colors.white]),
        ]))
        story.append(rt)
        story.append(Spacer(1, 0.5*cm))

        # Comparison table if both models
        if is_comparison and enet_probs:
            story.append(Paragraph("Model Comparison", h2_style))
            enet_top   = max(enet_probs, key=enet_probs.get)
            enet_conf  = enet_probs[enet_top]
            enet_level, _, _ = get_risk_level(enet_risk)
            comp_data = [
                ["Metric", "ViT-base (Main)", "EfficientNet-B3 (Baseline)"],
                ["Cancer Risk Score", f"{risk_score} / 100", f"{enet_risk:.1f} / 100"],
                ["Risk Level", level_text, enet_level],
                ["Most Likely", CLASS_INFO[top_cls]["name"], CLASS_INFO[enet_top]["name"]],
                ["Confidence", f"{confidence*100:.1f}%", f"{enet_conf*100:.1f}%"],
            ]
            ct = Table(comp_data, colWidths=[5*cm, 5.5*cm, 5.5*cm])
            ct.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,0), rl_colors.HexColor("#1D3A5F")),
                ("TEXTCOLOR",     (0,0), (-1,0), rl_colors.white),
                ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
                ("FONTNAME",      (0,1), (0,-1), "Helvetica-Bold"),
                ("BACKGROUND",    (0,1), (0,-1), rl_colors.HexColor("#E6F1FB")),
                ("FONTSIZE",      (0,0), (-1,-1), 10),
                ("GRID",          (0,0), (-1,-1), 0.5, rl_colors.HexColor("#CCCCCC")),
                ("LEFTPADDING",   (0,0), (-1,-1), 8),
                ("TOPPADDING",    (0,0), (-1,-1), 6),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                ("ROWBACKGROUNDS",(0,1), (-1,-1),
                 [rl_colors.white, rl_colors.HexColor("#F5F5F5")]),
                ("ALIGN", (1,0), (-1,-1), "CENTER"),
            ]))
            story.append(ct)
            story.append(Spacer(1, 0.5*cm))

        # Risk group summary (no classes column)
        story.append(Paragraph("Risk Group Summary", h2_style))
        group_data = [["Group", "Probability"]]
        group_colors_pdf = {
            "High Risk":   rl_colors.HexColor("#FCEBEB"),
            "Medium Risk": rl_colors.HexColor("#FAEEDA"),
            "Low Risk":    rl_colors.HexColor("#E1F5EE"),
        }
        group_text_colors = {
            "High Risk":   rl_colors.HexColor("#E24B4A"),
            "Medium Risk": rl_colors.HexColor("#EF9F27"),
            "Low Risk":    rl_colors.HexColor("#1D9E75"),
        }
        group_rows = []
        for gname, gclasses in [
            ("High Risk",   HIGH_RISK_C),
            ("Medium Risk", MEDIUM_RISK_C),
            ("Low Risk",    LOW_RISK_C),
        ]:
            gprob = sum(probs_dict.get(c, 0) for c in gclasses)
            group_data.append([gname, f"{gprob*100:.1f}%"])
            group_rows.append(gname)

        gt = Table(group_data, colWidths=[8*cm, 8*cm])
        gt_style = [
            ("BACKGROUND",    (0,0), (-1,0), rl_colors.HexColor("#1D3A5F")),
            ("TEXTCOLOR",     (0,0), (-1,0), rl_colors.white),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE",      (0,0), (-1,-1), 11),
            ("GRID",          (0,0), (-1,-1), 0.5, rl_colors.HexColor("#CCCCCC")),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("TOPPADDING",    (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("ALIGN",         (1,0), (1,-1), "CENTER"),
        ]
        for i, gname in enumerate(group_rows, 1):
            gt_style.append(("BACKGROUND", (0,i), (-1,i),
                            group_colors_pdf[gname]))
            gt_style.append(("TEXTCOLOR", (0,i), (0,i),
                            group_text_colors[gname]))
            gt_style.append(("FONTNAME", (0,i), (0,i), "Helvetica-Bold"))
        gt.setStyle(TableStyle(gt_style))
        story.append(gt)
        story.append(Spacer(1, 0.5*cm))

        # Detailed probabilities
        story.append(Paragraph("Detailed Class Probabilities", h2_style))
        prob_data = [["Class", "Probability", "Risk Weight"]]
        for cls, prob in sorted(probs_dict.items(),
                               key=lambda x: x[1], reverse=True):
            prob_data.append([
                CLASS_INFO[cls]["name"],
                f"{prob*100:.2f}%",
                f"{CLASS_INFO[cls]['weight']:.2f}"
            ])
        pt = Table(prob_data, colWidths=[8*cm, 4*cm, 4*cm])
        pt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), rl_colors.HexColor("#1D3A5F")),
            ("TEXTCOLOR",     (0,0), (-1,0), rl_colors.white),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE",      (0,0), (-1,-1), 10),
            ("GRID",          (0,0), (-1,-1), 0.5, rl_colors.HexColor("#CCCCCC")),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("ALIGN",         (1,0), (-1,-1), "CENTER"),
            ("ROWBACKGROUNDS",(0,1), (-1,-1),
             [rl_colors.white, rl_colors.HexColor("#F5F5F5")]),
        ]))
        story.append(pt)

        # ── PAGE 2, 3, 4 — Pie charts (one per page) ──────
        pie_groups = [
            ("High Risk",   HIGH_RISK_C,
             ["#E24B4A","#C0392B","#922B21"],   "#E24B4A"),
            ("Medium Risk", MEDIUM_RISK_C,
             ["#EF9F27","#F0B429","#F5C842"],   "#EF9F27"),
            ("Low Risk",    LOW_RISK_C,
             ["#1D9E75","#27AE60","#2ECC71","#58D68D"], "#1D9E75"),
        ]

        tmp_files = [tmp_scorebar.name]

        story.append(PageBreak())
        story.append(Paragraph("Risk Group Distribution", h2_style))
        story.append(Spacer(1, 0.3*cm))

        for gname, classes, colors, tcolor in pie_groups:
            gprob = sum(probs_dict.get(c, 0) for c in classes)

            story.append(Paragraph(
                f"{gname}  —  {gprob*100:.1f}%",
                ParagraphStyle("gh", fontSize=12, spaceAfter=6,
                              fontName="Helvetica-Bold",
                              textColor=rl_colors.HexColor(tcolor))
            ))

            # Pie chart
            fig_p, ax_p = plt.subplots(figsize=(4, 4))
            fig_p.patch.set_facecolor("white")
            ax_p.set_facecolor("white")
            vals = [max(probs_dict.get(c, 0), 0.001) for c in classes]
            wedges, texts, autotexts = ax_p.pie(
                vals, colors=colors,
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 10},
                wedgeprops={"edgecolor": "white", "linewidth": 2},
                pctdistance=0.75
            )
            for at in autotexts:
                at.set_fontsize(10)
                at.set_fontweight("bold")
            plt.tight_layout()
            tmp_p = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            plt.savefig(tmp_p.name, dpi=150, bbox_inches="tight",
                       facecolor="white")
            plt.close()
            tmp_files.append(tmp_p.name)

            # Layout: pie left, legend right
            legend_items = [Spacer(1, 0.3*cm)]
            for cls, color in zip(classes, colors):
                prob = probs_dict.get(cls, 0)
                legend_items.append(
                    Paragraph(
                        f"<font color='{color}'>■</font>  "
                        f"<b>{CLASS_INFO[cls]['name']}</b>  —  {prob*100:.1f}%",
                        ParagraphStyle("li", fontSize=11, fontName="Helvetica",
                                      spaceAfter=6,
                                      textColor=rl_colors.HexColor("#2C2C2A"))
                    )
                )

            row_table = Table(
                [[RLImage(tmp_p.name, width=5.5*cm, height=5.5*cm), legend_items]],
                colWidths=[6*cm, 10*cm]
            )
            row_table.setStyle(TableStyle([
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("LEFTPADDING", (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ]))
            story.append(row_table)
            story.append(Spacer(1, 0.1*cm))

        # ── PAGE 5 — Bar chart ────────────────────────────
        story.append(PageBreak())
        story.append(Paragraph("Class Probability Chart", h2_style))

        sorted_probs = sorted(probs_dict.items(),
                             key=lambda x: x[1], reverse=True)
        cls_names_bar = [CLASS_INFO[c]["name"] for c, _ in sorted_probs]
        cls_probs_bar = [p * 100 for _, p in sorted_probs]
        bar_colors_list = []
        for cls, _ in sorted_probs:
            if cls in HIGH_RISK_C:
                bar_colors_list.append("#E24B4A")
            elif cls in MEDIUM_RISK_C:
                bar_colors_list.append("#EF9F27")
            else:
                bar_colors_list.append("#1D9E75")

        fig_b, ax_b = plt.subplots(figsize=(12, 8))
        fig_b.patch.set_facecolor("white")
        ax_b.set_facecolor("white")
        bars_h = ax_b.barh(
            cls_names_bar[::-1], cls_probs_bar[::-1],
            color=bar_colors_list[::-1],
            edgecolor="white", height=0.6
        )
        ax_b.set_xlabel("Probability (%)", fontsize=13)
        ax_b.set_title("Class Probability Distribution",
                       fontsize=15, fontweight="bold", pad=15)
        ax_b.tick_params(axis="y", labelsize=12)
        ax_b.tick_params(axis="x", labelsize=11)
        for bar, prob in zip(bars_h, cls_probs_bar[::-1]):
            ax_b.text(bar.get_width() + 0.3,
                     bar.get_y() + bar.get_height()/2,
                     f"{prob:.1f}%", va="center", fontsize=11,
                     fontweight="bold")
        # Legend
        high_patch  = mpatches.Patch(color="#E24B4A", label="High Risk")
        med_patch   = mpatches.Patch(color="#EF9F27", label="Medium Risk")
        low_patch   = mpatches.Patch(color="#1D9E75", label="Low Risk")
        ax_b.legend(handles=[high_patch, med_patch, low_patch],
                   fontsize=11, loc="lower right")
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)
        plt.tight_layout()
        tmp_barchart = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        plt.savefig(tmp_barchart.name, dpi=150, bbox_inches="tight",
                   facecolor="white")
        plt.close()
        tmp_files.append(tmp_barchart.name)
        story.append(RLImage(tmp_barchart.name, width=16*cm, height=12*cm))

        # Disclaimer
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph(
            "Disclaimer: This report is for educational purposes only. "
            "It is not a medical diagnostic tool. "
            "Always consult a qualified dermatologist.",
            ParagraphStyle("disc", fontSize=9,
                          textColor=rl_colors.grey, fontName="Helvetica")
        ))

        doc.build(story)

        # Cleanup
        for f in tmp_files:
            try: os.unlink(f)
            except: pass
        buffer.seek(0)

        st.download_button(
            label="Download PDF Report",
            data=buffer,
            file_name="skin_cancer_risk_report.pdf",
            mime="application/pdf",
            key="download_pdf_main"
        )
        st.success("Report ready! Click above to download.")

# Footer
st.markdown("""
<div style="margin-top:48px; padding:24px 32px;
     border-top:1px solid #1E3A5F; text-align:center">
    <div style="display:flex; justify-content:center; align-items:center;
         gap:8px; margin-bottom:8px">
        <div style="width:6px; height:6px; background:#00B4D8;
             border-radius:50%"></div>
        <span style="color:white; font-size:15px; font-weight:700;
            letter-spacing:1px">DermAI</span>
        <div style="width:6px; height:6px; background:#00B4D8;
             border-radius:50%"></div>
    </div>
    <p style="color:#4A5568; font-size:12px; margin:0">
        GWU DATS 6303 &nbsp;·&nbsp; Group 4 &nbsp;·&nbsp; Spring 2026
        &nbsp;·&nbsp; For educational purposes only
    </p>
    <p style="color:#2A3548; font-size:11px; margin:6px 0 0 0">
        Powered by ViT-base-patch16-224 &nbsp;·&nbsp;
        HAM10000 + Supplementary Data &nbsp;·&nbsp;
        13,215 training images &nbsp;·&nbsp; 10 classes
    </p>
</div>
""", unsafe_allow_html=True)

# Demo mode image loader
if "demo_mode" in dir() and demo_mode and selected_demo:
    demo_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        demo_images[selected_demo]
    )
    if os.path.exists(demo_path):
        st.sidebar.image(
            Image.open(demo_path),
            caption=selected_demo,
            width=180
        )
        st.sidebar.info("Upload this image manually to analyse it.")
