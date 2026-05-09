import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Full Analysis",
    layout="wide"
)

# ================= HEADER =================

st.title("🖼 Full Detection Analysis")

st.caption(
    "Detailed YOLOv8-OBB inference visualization and system diagnostics."
)

# ================= CHECK SESSION =================

if "analysis_image" not in st.session_state:

    st.warning("No analysis data found.")
    st.stop()

# ================= LOAD DATA =================

annotated = st.session_state["analysis_image"]

detected_label = st.session_state["detected_label"]

detected_conf = st.session_state["detected_conf"]

inference_time = st.session_state["inference_time"]

# ================= MAIN IMAGE =================

st.image(
    annotated,
    caption="YOLOv8 OBB Full Prediction Output",
    use_container_width=True
)

# ================= METRICS =================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Detected Label",
        detected_label.upper()
    )

with col2:
    st.metric(
        "Confidence",
        f"{detected_conf:.2f}"
    )

with col3:
    st.metric(
        "Inference Time",
        f"{inference_time:.2f} ms"
    )

# ================= LOGS =================

st.markdown("## 📜 System Logs")

logs = {
    "detected_label": detected_label,
    "confidence": detected_conf,
    "inference_time_ms": inference_time,
    "model": "YOLOv8 OBB",
    "framework": "Ultralytics",
    "device": "CPU"
}

st.json(logs)

# ================= BACK BUTTON =================

if st.button("⬅ Back to Dashboard"):

    st.switch_page("app.py")
