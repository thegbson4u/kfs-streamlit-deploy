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

# ================= MAIN IMAGE =================

st.markdown("""
<style>

.bento-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 18px;
    padding: 24px;
    margin-top: 20px;
    margin-bottom: 25px;
    box-shadow: 0 4px 25px rgba(0,0,0,0.25);
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="bento-card">',
    unsafe_allow_html=True
)

# Centered Layout
col1, col2, col3 = st.columns([1, 3, 1])

with col2:

    st.image(
        annotated,
        caption="YOLOv8 OBB Full Prediction Output",
        width=850
    )

st.markdown(
    '</div>',
    unsafe_allow_html=True
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
