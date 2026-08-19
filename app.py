import streamlit as st
import cv2
import time
from ultralytics import YOLO

st.set_page_config(
    page_title="AegisVision Tactical Command Center",
    page_icon="🛡️",
    layout="wide"
)

# Advanced Military HUD CSS Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #05070b;
        color: #00ffcc;
        font-family: 'Courier New', Courier, monospace;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00ffcc !important;
        font-family: 'Courier New', Courier, monospace;
        letter-spacing: 1.5px;
    }
    .live-badge {
        background-color: #ff0033;
        color: white;
        padding: 5px 12px;
        border-radius: 2px;
        font-weight: bold;
        font-size: 13px;
        letter-spacing: 2px;
        box-shadow: 0 0 10px #ff0033;
    }
    div[data-testid="stMetricValue"] {
        color: #00ffcc !important;
        font-family: 'Courier New', Courier, monospace;
        font-size: 28px !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #8a99ad !important;
        font-size: 12px !important;
        letter-spacing: 1px;
    }
    .stButton>button {
        background-color: #00ffcc22;
        color: #00ffcc;
        border: 1px solid #00ffcc;
        border-radius: 2px;
        font-weight: bold;
        letter-spacing: 1px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #00ffcc;
        color: #05070b;
        box-shadow: 0 0 15px #00ffcc;
    }
    </style>
""", unsafe_allow_html=True)

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Top Bar Header
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.title("🛡️ AEGIS-VI: BATTLEFIELD COMMAND & TRACKING")
with col_badge:
    st.markdown("<br><span class='live-badge'>● LIVE FEED</span>", unsafe_allow_html=True)

st.markdown("---")

# Main Layout
col1, col2 = st.columns([2.3, 1])

with col1:
    st.markdown("### 🎥 PRIMARY SURVEILLANCE FEED (SECTOR 7)")
    video_placeholder = st.empty()

with col2:
    st.markdown("### 📊 SYSTEM TELEMETRY")

    m1, m2 = st.columns(2)
    metric_latency = m1.empty()
    metric_locked = m2.empty()
    
    metric_latency.metric(label="LATENCY", value="0.0 ms")
    metric_locked.metric(label="TARGETS LOCKED", value="0")

    st.markdown("### 📡 ACTIVE TARGET LOGS")
    logs_placeholder = st.empty()

# Video Processing Loop
video_path = "170300-843059179.mp4"
cap = cv2.VideoCapture(video_path)

if st.button("🚀 ENGAGE TACTICAL STREAM"):
    if not cap.isOpened():
        st.error(f"Error: Could not open video file {video_path}.")
    else:
        while cap.isOpened():
            t_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Run standard YOLO inference
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()

            # Compute real-time metrics
            t_elapsed = (time.perf_counter() - t_start) * 1000
            metric_latency.metric(label="LATENCY", value=f"{t_elapsed:.1f} ms", delta="OPTIMIZED")

            target_count = len(results[0].boxes)
            metric_locked.metric(label="TARGETS LOCKED", value=f"{target_count}")

            # Generate target logs
            log_text = ""
            if target_count > 0:
                for idx, box in enumerate(results[0].boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0]) * 100
                    label_name = model.names[cls_id].upper()
                    log_text += f">> [ID: T-10{idx+1}] {label_name} | CONF: {conf:.1f}% | STATUS: TRACKING\n"
            else:
                log_text = ">> SCANNING SECTOR... NO THREATS DETECTED."

            logs_placeholder.code(log_text)

            # Stream frame to UI
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

cap.release()
