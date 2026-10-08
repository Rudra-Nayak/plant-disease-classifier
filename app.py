"""
Leaf Disease Classifier - Botanical Field Guide UI
Clean, editorial aesthetic with warm paper palette, serif headings,
and zero artificial decoration.
"""

import json
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Leaf Disease Check",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS - Botanical Field Notebook Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400&family=Source+Sans+3:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

/* ---- Base Document Canvas ---- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #F4F1EA !important;
    color: #1F3A2E !important;
    font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

[data-testid="stAppViewContainer"] {
    background: #F4F1EA !important;
}

[data-testid="stHeader"] {
    background-color: transparent !important;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1040px;
    padding-top: 1.25rem;
    padding-bottom: 2.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* ---- Typography ---- */
h1, h2, h3, h4, .serif-heading {
    font-family: 'Newsreader', Georgia, serif !important;
    color: #1F3A2E !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}

p, span, label, div {
    color: #1F3A2E;
}

.text-muted {
    color: #5B6B62 !important;
}

.field-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #5B6B62;
    margin-bottom: 0.35rem;
}

/* ---- Hide Default Streamlit Chrome ---- */
[data-testid="stToolbarActions"],
.stAppDeployButton,
#MainMenu,
footer,
[data-testid="stStatusWidget"] {
    display: none !important;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background-color: #EAE5DB !important;
    border-right: 1px solid #D9D3C5 !important;
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] hr {
    border-color: #D9D3C5 !important;
    margin: 1.25rem 0 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #1F3A2E;
    font-size: 0.9rem;
    line-height: 1.45;
}

/* ---- Tabs ---- */
div[data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid #D9D3C5 !important;
    gap: 1.5rem !important;
    padding: 0 !important;
    margin-bottom: 1rem !important;
}

button[data-baseweb="tab"] {
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    color: #5B6B62 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 0.5rem 0.25rem !important;
}

button[data-baseweb="tab"]:hover {
    color: #1F3A2E !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #1F3A2E !important;
    border-bottom: 2px solid #C2542D !important;
    font-weight: 600 !important;
}

div[data-baseweb="tab-highlight"],
div[data-baseweb="tab-border"] {
    display: none !important;
}

/* ---- File Uploader & Dropzone Restyling ---- */
[data-testid="stFileUploader"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}

[data-testid="stFileUploader"] [data-testid="stWidgetLabel"] {
    display: none !important;
}

[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"],
section[data-testid="stFileUploaderDropzone"] {
    background-color: #EAE5DB !important;
    border: 1px dashed #D9D3C5 !important;
    border-radius: 4px !important;
    padding: 1.25rem 1rem !important;
    box-shadow: none !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.5rem !important;
}

[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:hover,
section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #C2542D !important;
    background-color: #ECE7DD !important;
}

[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] div {
    color: #5B6B62 !important;
    font-family: 'Source Sans 3', sans-serif !important;
}

[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] svg {
    stroke: #5B6B62 !important;
    fill: none !important;
    color: #5B6B62 !important;
}

[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] button {
    background-color: #C2542D !important;
    color: #FFFFFF !important;
    border: 1px solid #A43F1B !important;
    border-radius: 4px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.35rem 0.9rem !important;
    box-shadow: none !important;
}

[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #A43F1B !important;
}

/* Uploaded file preview row */
[data-testid="stFileUploaderFileData"],
div[data-testid="stFileUploaderFileData"] {
    background-color: #EAE5DB !important;
    border: 1px solid #D9D3C5 !important;
    border-radius: 4px !important;
    padding: 0.5rem 0.75rem !important;
    margin-top: 0.6rem !important;
}

[data-testid="stFileUploaderFileData"] span,
[data-testid="stFileUploaderFileData"] small,
[data-testid="stFileUploaderFileData"] div {
    color: #1F3A2E !important;
    font-family: 'Source Sans 3', sans-serif !important;
}

[data-testid="stFileUploaderFileData"] button {
    background-color: transparent !important;
    border: none !important;
    color: #C2542D !important;
    box-shadow: none !important;
}

[data-testid="stFileUploaderFileData"] button:hover {
    background-color: rgba(194, 84, 45, 0.1) !important;
}

[data-testid="stFileUploaderFileData"] button svg {
    stroke: #C2542D !important;
    color: #C2542D !important;
}

/* Camera */
[data-testid="stCameraInput"] {
    background-color: #EAE5DB !important;
    border: 1px dashed #D9D3C5 !important;
    border-radius: 4px !important;
    padding: 0.75rem !important;
}

/* ---- Image Display ---- */
[data-testid="stImage"] img {
    border: 1px solid #D9D3C5 !important;
    border-radius: 4px !important;
    max-height: 380px !important;
    width: 100% !important;
    object-fit: contain !important;
    background-color: #EAE5DB;
}

/* ---- Result Panel ---- */
.result-panel {
    border: 1px solid #D9D3C5;
    background-color: #EAE5DB;
    border-radius: 4px;
    padding: 1.5rem;
}

.result-panel-healthy {
    border-left: 4px solid #2D6A4F;
}

.result-panel-diseased {
    border-left: 4px solid #C2542D;
}

.status-line {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.5rem;
}

.status-healthy {
    color: #2D6A4F;
}

.status-diseased {
    color: #C2542D;
}

.metric-bar-container {
    margin: 1.25rem 0 1.5rem 0;
    padding: 0.75rem 0;
    border-top: 1px solid #D9D3C5;
    border-bottom: 1px solid #D9D3C5;
}

.bar-track {
    width: 100%;
    height: 6px;
    background-color: #D9D3C5;
    border-radius: 3px;
    overflow: hidden;
    margin-top: 0.4rem;
}

.bar-fill {
    height: 100%;
    background-color: #C2542D;
    border-radius: 3px;
}

.alternative-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.4rem 0;
    border-bottom: 1px solid #D9D3C5;
    font-size: 0.88rem;
}

.alternative-row:last-child {
    border-bottom: none;
}

.note-box {
    border: 1px solid #D9D3C5;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin-top: 1.25rem;
    background-color: #F4F1EA;
    font-size: 0.82rem;
    line-height: 1.45;
    color: #5B6B62;
}

/* ---- Mobile Responsiveness ---- */
@media (max-width: 768px) {
    [data-testid="stMainBlockContainer"] {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Model loading (cached on CPU)
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("classes.json") as f:
        classes = json.load(f)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(
        torch.load("plant_model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    return model, classes


@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def pretty_name(raw: str) -> tuple:
    """Format raw dataset folder name into readable plant name and condition."""
    parts = raw.split("___")
    plant = parts[0].replace("_", " ").strip()
    if len(parts) > 1:
        cond = parts[1].strip("_").replace("_", " ").strip().title()
    else:
        cond = ""
    plant = " ".join(plant.split())
    cond = " ".join(cond.split())
    return plant, cond


def predict(img, model, classes, tfm, k=3):
    inp = tfm(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(inp), 1).squeeze(0)
    topk = probs.topk(k)
    return [(classes[i], p) for p, i in
            zip(topk.values.tolist(), topk.indices.tolist())]


def format_result_card(plant: str, condition: str, is_healthy: bool,
                       top_conf: float, preds: list) -> str:
    """Construct clean HTML string without markdown-code-block indentation."""
    border_class = "result-panel-healthy" if is_healthy else "result-panel-diseased"
    status_text = "Status: Healthy foliage" if is_healthy else "Status: Pathogen / Disease detected"
    status_class = "status-healthy" if is_healthy else "status-diseased"

    rows = []
    for cls_name, conf_val in preds:
        alt_plant, alt_cond = pretty_name(cls_name)
        pct = conf_val * 100
        rows.append(
            f'<div class="alternative-row">'
            f'<span>{alt_plant} - {alt_cond}</span>'
            f'<span style="font-weight:600; color:#5B6B62;">{pct:.1f}%</span>'
            f'</div>'
        )
    alternatives_html = "".join(rows)

    low_conf_html = ""
    if top_conf < 0.60:
        low_conf_html = (
            '<div class="note-box" style="border-left: 3px solid #C2542D; margin-top:1rem;">'
            '<strong>Low diagnostic confidence (under 60%).</strong><br>'
            'The model cannot make a definitive match. Check that the leaf belongs to one of the '
            '14 supported crops and is clearly focused against an uncluttered background.'
            '</div>'
        )

    card = (
        f'<div class="result-panel {border_class}">'
        f'<div class="field-label">{plant}</div>'
        f'<h2 style="font-size:1.85rem; margin:0.15rem 0 0.5rem 0; line-height:1.2;">{condition}</h2>'
        f'<div class="status-line {status_class}">{status_text}</div>'
        f'<div class="metric-bar-container">'
        f'<div style="display:flex; justify-content:space-between; align-items:baseline;">'
        f'<span style="font-size:0.86rem; color:#5B6B62;">Diagnostic confidence</span>'
        f'<span style="font-size:1.25rem; font-weight:700; color:#1F3A2E;">{top_conf * 100:.1f}%</span>'
        f'</div>'
        f'<div class="bar-track">'
        f'<div class="bar-fill" style="width: {max(top_conf * 100, 3):.1f}%;"></div>'
        f'</div>'
        f'</div>'
        f'<div class="field-label" style="margin-bottom:0.6rem;">Top differential classifications</div>'
        f'{alternatives_html}'
        f'{low_conf_html}'
        f'<div class="note-box">'
        f'<strong>Limitation note:</strong> This model is strictly trained on 14 crops. '
        f'Unrelated botanical species (such as ornamental flowers or trees) may yield confident '
        f'yet incorrect matches.'
        f'</div>'
        f'</div>'
    )
    return card


# ──────────────────────────────────────────────
# Sidebar - Field Notes & Limitations
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="field-label">Reference Guide</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-size:1.35rem; margin-top:0.2rem; margin-bottom:1rem;">Leaf Disease Index</h2>', unsafe_allow_html=True)

    st.markdown('<div class="field-label">Supported crops</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.86rem; color:#1F3A2E; margin-bottom:1.5rem; line-height:1.5;">'
        'apple, blueberry, cherry, corn, grape, orange, peach, pepper, potato, '
        'raspberry, soybean, squash, strawberry, tomato'
        '</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="field-label">Limitations</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.84rem; color:#5B6B62; margin-bottom:1.5rem; line-height:1.45;">'
        'Trained on PlantVillage (lab-style leaf photos). Plants outside the 14 crops '
        'can receive confident but wrong predictions.'
        '</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="field-label">Model details</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.82rem; color:#5B6B62; line-height:1.45;">'
        'ResNet-18 architecture. Evaluates 38 distinct plant-pathogen classifications on CPU.'
        '</p>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Main Header (Left-aligned, restrained)
# ──────────────────────────────────────────────
st.markdown(
    '<div style="margin-bottom: 1.5rem;">'
    '<div class="field-label">Botanical Diagnosis</div>'
    '<h1 style="font-size:2.2rem; margin: 0 0 0.3rem 0;">Leaf disease check</h1>'
    '<p class="text-muted" style="font-size:0.98rem; margin:0; line-height:1.5;">'
    'Upload a photo of a single leaf on a plain background.'
    '</p>'
    '</div>',
    unsafe_allow_html=True,
)

# Load model
try:
    model, classes = load_model()
    transform = get_transform()
except Exception as exc:
    st.error(
        f"Unable to load model weights (plant_model.pt or classes.json): {exc}"
    )
    st.stop()


# ──────────────────────────────────────────────
# Desktop Two-Column Layout
# ──────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="field-label">Input Specimen</div>', unsafe_allow_html=True)

    tab_upload, tab_camera = st.tabs(["File Upload", "Camera"])
    image = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Select leaf photograph",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            try:
                image = Image.open(uploaded).convert("RGB")
            except (UnidentifiedImageError, Exception):
                image = None
                st.error("Unreadable file format. Please upload a standard photograph (JPG, PNG, or WebP).")

    with tab_camera:
        cam = st.camera_input("Photograph specimen", label_visibility="collapsed")
        if cam:
            try:
                image = Image.open(cam).convert("RGB")
            except (UnidentifiedImageError, Exception):
                image = None
                st.error("Unable to process camera image. Please try again.")

    if image is not None:
        st.markdown('<div class="field-label" style="margin-top:1rem;">Specimen Preview</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

with col_right:
    st.markdown('<div class="field-label">Diagnosis & Findings</div>', unsafe_allow_html=True)

    if image is None:
        st.markdown(
            '<div class="result-panel" style="border-left: 1px solid #D9D3C5;">'
            '<p class="text-muted" style="margin:0; font-size:0.92rem; line-height:1.5;">'
            'Awaiting specimen. Once uploaded, the classification and confidence evaluation '
            'will appear here directly alongside the image.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        w, h = image.size
        if w < 32 or h < 32:
            st.error("Image resolution is insufficient for pathology analysis. Please upload a larger image.")
        else:
            with st.spinner("Classifying specimen..."):
                preds = predict(image, model, classes, transform, k=3)

            top_cls, top_conf = preds[0]
            plant, condition = pretty_name(top_cls)
            is_healthy = condition.lower() == "healthy"

            # Render HTML without markdown code-block indentation
            result_html = format_result_card(plant, condition, is_healthy, top_conf, preds)
            st.markdown(result_html, unsafe_allow_html=True)
