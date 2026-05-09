import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
from PIL import Image

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="KFS Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

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

# ==================== CUSTOM CSS ====================

st.markdown("""
<style>

/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main App */
.stApp {
    background-color: #0D1117;
    color: #E6EDF3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #11161D;
    border-right: 1px solid #30363D;
}

/* Header */
.main-title {
    font-size: 68px;
    font-weight: 900;
    color: white;
    margin-bottom: 8px;
    letter-spacing: -1px;
}

.sub-title {
    font-size: 26px;
    color: #9BA3AF;
    margin-bottom: 40px;
    font-weight: 400;
}

/* Bento Cards */
.bento-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
    transition: 0.2s ease;
    box-shadow: 0 4px 25px rgba(0,0,0,0.25);
}

.bento-card:hover {
    transform: translateY(-3px);
}

/* Results */
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

/* Upload Box */
[data-testid="stFileUploader"] {
    border: 2px dashed #007BFF;
    border-radius: 18px;
    padding: 18px;
    background: #161B22;
}

/* Buttons */
.stButton > button {
    background-color: #007BFF;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 18px;
    font-weight: 600;
    width: 100%;
}

.stButton > button:hover {
    background-color: #0066D6;
    transform: translateY(-1px);
}

/* Metrics */
[data-testid="metric-container"] {
    background: #161B22;
    border: 1px solid #30363D;
    padding: 18px;
    border-radius: 16px;
}

</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================

st.markdown(
    '<p class="main-title">🔍 KFS Detection Dashboard</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Real vs Fake KFS Detection using YOLOv8-OBB</p>',
    unsafe_allow_html=True
)

# ==================== LOAD MODEL ====================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ==================== SIDEBAR ====================

with st.sidebar:

    st.markdown("# ⚙ Dashboard Settings")

    with st.expander("📌 Model Labels", expanded=False):

        for idx, name in model.names.items():
            st.write(f"Class {idx}: {name}")

    with st.expander("🎚 Detection Settings", expanded=True):

        conf = st.slider(
            "Confidence Threshold",
            0.1,
            0.9,
            0.3,
            0.05
        )

    with st.expander("🧠 Model Metadata", expanded=False):

        st.markdown("""
        **Architecture:** YOLOv8 OBB  
        **Input Size:** 640  
        **Device:** CPU  
        **Framework:** Ultralytics  
        **Classes:** 2  
        """)

    st.markdown("---")

    st.caption(
        "Built with YOLOv8-OBB, Streamlit, OpenCV and custom dataset training."
    )

# ==================== FILE UPLOADER ====================

uploaded_file = st.file_uploader(
    "📂 Drag & Drop Image Here",
    type=["jpg", "jpeg", "png"]
)

# ==================== MAIN PROCESS ====================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # ==================== INFERENCE ====================

    start_time = time.time()

    results = model.predict(
        source=image_np,
        imgsz=IMG_SIZE,
        conf=conf,
        device=DEVICE,
        verbose=False
    )

    inference_time = (time.time() - start_time) * 1000

    annotated = image_np.copy()

    h, w, _ = annotated.shape

    detected_label = "No Detection"
    detected_conf = 0.0

    # ==================== DRAW DETECTIONS ====================

    for r in results:

        if r.obb is None:
            continue

        boxes = r.obb.xyxyxyxy.cpu().numpy()
        classes = r.obb.cls.cpu().numpy()
        scores = r.obb.conf.cpu().numpy()

        for box, cls, score in zip(boxes, classes, scores):

            pts = box.reshape(-1, 2).astype(int)

            # Draw Bounding Box
            cv2.polylines(
                annotated,
                [pts],
                isClosed=True,
                color=BOX_COLOR,
                thickness=BOX_THICKNESS
            )

            label = model.names[int(cls)]

            detected_label = label
            detected_conf = score

            text = f"{label.upper()} {score:.2f}"

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

            # White Rectangle
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

    # ==================== DASHBOARD GRID ====================

    col1, col2 = st.columns([1, 1])

    # ==================== RESULT CARD ====================

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

        st.progress(float(detected_conf))

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================== REDIRECT BUTTON ====================

        if st.button(
            "🔍 Open Full Analysis",
            use_container_width=True
        ):

            st.session_state["analysis_image"] = annotated
            st.session_state["detected_label"] = detected_label
            st.session_state["detected_conf"] = float(detected_conf)
            st.session_state["inference_time"] = inference_time

            st.switch_page("Full_Analysis.py")

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # ==================== PREVIEW CARD ====================

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

    # ==================== METADATA GRID ====================

    col3, col4 = st.columns([1, 1])

    # ==================== PERFORMANCE ====================

    with col3:

        st.markdown(
            '<div class="bento-card">',
            unsafe_allow_html=True
        )

        st.markdown("## ⚡ Performance Metrics")

        st.metric(
            "Inference Time",
            f"{inference_time:.2f} ms"
        )

        st.metric(
            "Model",
            "YOLOv8 OBB"
        )

        st.metric(
            "Device",
            DEVICE.upper()
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # ==================== SYSTEM LOGS ====================

    with col4:

        st.markdown(
            '<div class="bento-card">',
            unsafe_allow_html=True
        )

        st.markdown("## 📜 System Logs")

        detection_logs = {
            "detected_label": detected_label,
            "confidence": float(detected_conf),
            "inference_time_ms": round(inference_time, 2),
            "device": DEVICE,
            "image_size": IMG_SIZE,
            "model": "YOLOv8 OBB"
        }

        st.json(detection_logs)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

