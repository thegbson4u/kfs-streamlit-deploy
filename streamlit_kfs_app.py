import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ==================== CONFIG (CHANGE HERE ONLY) ====================
MODEL_PATH ="kfs_obb_v1.pt"

IMG_SIZE = 640
DEVICE = "cpu"  # 0 = GPU, "cpu" if needed

# Bounding box style
BOX_COLOR = (0, 255, 0)     # Green (BGR)
BOX_THICKNESS = 5

# Label style (HIGH VISIBILITY)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.6            # 🔴 BIG TEXT (increase if needed)
FONT_THICKNESS = 3

LABEL_BG_COLOR = (255, 255, 255)   # WHITE background
LABEL_TEXT_COLOR = (0, 0, 0)       # BLACK text
LABEL_PADDING = 10
# ==================================================================

st.set_page_config(page_title="KFS Detection (OBB)", layout="centered")
st.title("🔍 KFS Detection System (OBB)")
st.write("High-accuracy **Real vs Fake KFS** detection using YOLOv8-OBB")

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

conf = st.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # -------------------- INFERENCE --------------------
    results = model.predict(
        source=image_np,
        imgsz=IMG_SIZE,
        conf=conf,
        device=DEVICE,
        verbose=False
    )

    annotated = image_np.copy()
    h, w, _ = annotated.shape

    for r in results:
        if r.obb is None:
            continue

        boxes = r.obb.xyxyxyxy.cpu().numpy()
        classes = r.obb.cls.cpu().numpy()
        scores = r.obb.conf.cpu().numpy()

        for box, cls, score in zip(boxes, classes, scores):
            pts = box.reshape(-1, 2).astype(int)

            # ---- Draw rotated bounding box ----
            cv2.polylines(
                annotated,
                [pts],
                isClosed=True,
                color=BOX_COLOR,
                thickness=BOX_THICKNESS
            )

            # ---- Prepare label text ----
            label = model.names[int(cls)]
            text = f"{label.upper()}  {score:.2f}"

            (text_w, text_h), _ = cv2.getTextSize(
                text, FONT, FONT_SCALE, FONT_THICKNESS
            )

            # ---- Label position (safe inside image) ----
            x0, y0 = pts[0]
            x0 = max(0, min(x0, w - text_w - LABEL_PADDING * 2))
            y0 = max(text_h + LABEL_PADDING * 2, y0)

            # ---- Draw WHITE filled rectangle ----
            cv2.rectangle(
                annotated,
                (x0, y0 - text_h - LABEL_PADDING * 2),
                (x0 + text_w + LABEL_PADDING * 2, y0),
                LABEL_BG_COLOR,
                -1
            )

            # ---- Draw BLACK text ----
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

    # -------------------- DISPLAY --------------------
    st.image(annotated, caption="Prediction Result", width=700)


