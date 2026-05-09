import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ==================== CONFIG ====================

MODEL_PATH = "kfs_obb_v1.pt"

IMG_SIZE = 640
DEVICE = "cpu"

# Bounding Box Style
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2

# Label Style
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2

LABEL_BG_COLOR = (255, 255, 255)
LABEL_TEXT_COLOR = (0, 0, 0)
LABEL_PADDING = 6

# =================================================

st.set_page_config(
    page_title="KFS Detection System",
    layout="wide"
)

# ================= CUSTOM CSS ====================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

/* Header */
.title-text {
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 0px;
}

.subtitle-text {
    color: #9BA3AF;
    font-size: 18px;
    margin-top: -10px;
    margin-bottom: 30px;
}

/* Bento Cards */
.bento-card {
    background: #161B22;
    padding: 24px;
    border-radius: 24px;
    border: 1px solid #2A2F3A;
    margin-bottom: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.25);
}

/* Result Text */
.real-result {
    color: #22C55E;
    font-size: 34px;
    font-weight: 800;
}

.fake-result {
    color: #EF4444;
    font-size: 34px;
    font-weight: 800;
}

.metric-label {
    color: #9BA3AF;
    font-size: 14px;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER ====================

st.markdown(
    '<p class="title-text">🔍 KFS Detection System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle-text">High-accuracy Real vs Fake KFS detection using YOLOv8-OBB</p>',
    unsafe_allow_html=True
)

# ================= LOAD MODEL ====================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ================= LABELS ====================

st.markdown("### 📌 Model Labels")

for idx, name in model.names.items():
    st.write(f"Class {idx}: {name}")

# ================= SETTINGS ====================

conf = st.slider(
    "Confidence Threshold",
    0.1,
    0.9,
    0.3,
    0.05
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# ================= PROCESS IMAGE ====================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # ================= INFERENCE ====================

    results = model.predict(
        source=image_np,
        imgsz=IMG_SIZE,
        conf=conf,
        device=DEVICE,
        verbose=False
    )

    annotated = image_np.copy()

    h, w, _ = annotated.shape

    detected_label = "No Detection"
    detected_conf = 0.0

    # ================= DRAW DETECTIONS ====================

    for r in results:

        if r.obb is None:
            continue

        boxes = r.obb.xyxyxyxy.cpu().numpy()
        classes = r.obb.cls.cpu().numpy()
        scores = r.obb.conf.cpu().numpy()

        for box, cls, score in zip(boxes, classes, scores):

            pts = box.reshape(-1, 2).astype(int)

            # Draw OBB
            cv2.polylines(
                annotated,
                [pts],
                isClosed=True,
                color=BOX_COLOR,
                thickness=BOX_THICKNESS
            )

            # Label
            label = model.names[int(cls)]

            detected_label = label
            detected_conf = score

            text = f"{label.upper()}  {score:.2f}"

            (text_w, text_h), _ = cv2.getTextSize(
                text,
                FONT,
                FONT_SCALE,
                FONT_THICKNESS
            )

            x0, y0 = pts[0]

            x0 = max(
                0,
                min(x0, w - text_w - LABEL_PADDING * 2)
            )

            y0 = max(
                text_h + LABEL_PADDING * 2,
                y0
            )

            # White Background
            cv2.rectangle(
                annotated,
                (x0, y0 - text_h - LABEL_PADDING * 2),
                (x0 + text_w + LABEL_PADDING * 2, y0),
                LABEL_BG_COLOR,
                -1
            )

            # Text
            cv2.putText(
                annotated,
                text,
                (x0 + LABEL_PADDING, y0 - LABEL_PADDING),
                FONT,
                FONT_SCALE,
                LABEL_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA
            )

    # ================= BENTO GRID ====================

    col1, col2 = st.columns([1, 1])

    # ================= RESULT CARD ====================

    with col1:

        st.markdown(
            '<div class="bento-card">',
            unsafe_allow_html=True
        )

        st.markdown("## 📋 Detection Result")

        if detected_label == "real kfs":

            st.markdown(
                '<p class="real-result">✅ REAL KFS</p>',
                unsafe_allow_html=True
            )

        elif detected_label == "fake kfs":

            st.markdown(
                '<p class="fake-result">⚠️ FAKE KFS</p>',
                unsafe_allow_html=True
            )

        else:
            st.warning("No Detection")

        st.metric(
            label="Confidence Score",
            value=f"{detected_conf:.2f}"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # ================= PREVIEW CARD ====================

    with col2:

        st.markdown(
            '<div class="bento-card">',
            unsafe_allow_html=True
        )

        st.markdown("## 🎯 Detection Preview")

        preview = cv2.resize(
            annotated,
            (320, 320)
        )

        st.image(
            preview,
            caption="Highlighted Detection",
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # ================= FULL IMAGE ====================

    st.markdown("## 🖼 Full Detection Analysis")

    st.image(
        annotated,
        caption="YOLOv8 OBB Prediction Output",
        use_container_width=True
    )

