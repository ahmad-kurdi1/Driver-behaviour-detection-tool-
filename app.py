"""
============================================================================
Driver Behavior Detection Tool — Streamlit Web Interface
Supports: Live Camera, Upload Video, Upload Image
YOLO11s + dlib Facial Landmark Analysis
============================================================================
"""

import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from pathlib import Path
from collections import deque
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image

# ============================================================
# COCO to Driver Behavior Mapping
# ============================================================
COCO_TO_DRIVER = {
    67: {'class_name': 'Talking on Phone / Texting', 'color': (0, 140, 255), 'behavior': 'distracted'},
    41: {'class_name': 'Drinking', 'color': (0, 0, 255), 'behavior': 'distracted'},
    39: {'class_name': 'Drinking', 'color': (0, 0, 255), 'behavior': 'distracted'},
    40: {'class_name': 'Drinking', 'color': (0, 0, 255), 'behavior': 'distracted'},
    45: {'class_name': 'Eating / Drinking', 'color': (0, 0, 255), 'behavior': 'distracted'},
    46: {'class_name': 'Eating', 'color': (255, 0, 255), 'behavior': 'distracted'},
    52: {'class_name': 'Eating', 'color': (255, 0, 255), 'behavior': 'distracted'},
    53: {'class_name': 'Eating', 'color': (255, 0, 255), 'behavior': 'distracted'},
    73: {'class_name': 'Reading / Distracted', 'color': (255, 255, 0), 'behavior': 'distracted'},
    76: {'class_name': 'Distracted (Object in Hand)', 'color': (255, 255, 0), 'behavior': 'distracted'},
}

BEHAVIOR_COLORS = {
    'Talking on Phone / Texting': '#E67E22',
    'Drinking': '#E74C3C',
    'Eating / Drinking': '#E74C3C',
    'Eating': '#9B59B6',
    'Reading / Distracted': '#F1C40F',
    'Distracted (Object in Hand)': '#F1C40F',
}


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Driver Behavior Detection Tool",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .title-bar {
        background: linear-gradient(135deg, #1B4F72 0%, #2E86C1 100%);
        padding: 1rem 2rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .title-bar h1 { margin: 0; font-size: 1.6rem; font-weight: 700; }
    .title-bar .subtitle { font-size: 0.85rem; opacity: 0.9; }
    
    .status-badge {
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        text-align: center;
        min-width: 140px;
    }
    .status-safe { background: #D5F5E3; color: #1E8449; border: 2px solid #27AE60; }
    .status-warning { background: #FCF3CF; color: #B7950B; border: 2px solid #F1C40F; animation: pulse 1.5s infinite; }
    .status-danger { background: #FADBD8; color: #C0392B; border: 2px solid #E74C3C; animation: pulse 0.8s infinite; }
    .status-idle { background: #EBF5FB; color: #2471A3; border: 2px solid #3498DB; }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.03); }
    }
    
    .control-panel {
        background: #F8F9F9;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid #E5E8E8;
    }
    
    .metric-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .metric-mini {
        flex: 1;
        min-width: 60px;
        background: #F8F9F9;
        border-radius: 8px;
        padding: 8px;
        text-align: center;
        border: 1px solid #E5E8E8;
    }
    .metric-mini .value { font-size: 1.3rem; font-weight: 700; color: #2C3E50; }
    .metric-mini .label { font-size: 0.65rem; color: #7F8C8D; text-transform: uppercase; }
    
    .block-container { padding-top: 0.5rem; padding-bottom: 0; }
    .stButton button { padding: 0.3rem 1rem; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Load Models (Cached)
# ============================================================
@st.cache_resource
def load_yolo_model(model_choice="YOLOv8s (Stable)"):
    """Load YOLO model based on user choice"""
    from ultralytics import YOLO
    
    if model_choice == "YOLOv8s (Stable)":
        model_path = Path(__file__).parent / 'models' / 'driver_behavior' / 'weights' / 'best.pt'
    elif model_choice == "YOLO11s (New)":
        model_path = Path(__file__).parent / 'models' / 'driver_behavior_roboflow' / 'weights' / 'best.pt'
    else:
        model_path = Path(__file__).parent / 'models' / 'driver_behavior' / 'weights' / 'best.pt'
    
    if not model_path.exists():
        model_path = 'yolo11s.pt'
    
    model = YOLO('yolo11s.pt')
    class_names = model.names
    return model, class_names, str(model_path)


@st.cache_resource
def load_dlib():
    """Load dlib face detector and predictor"""
    import dlib
    landmark_path = Path(__file__).parent / 'utils' / 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(landmark_path)) if landmark_path.exists() else None
    return detector, predictor


def calculate_ear(landmarks, eye_indices):
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in eye_indices]
    points = [np.array(p) for p in points]
    v1 = np.linalg.norm(points[1] - points[5])
    v2 = np.linalg.norm(points[2] - points[4])
    h = np.linalg.norm(points[0] - points[3])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0


def calculate_head_pose(landmarks):
    nose = np.array([landmarks.part(30).x, landmarks.part(30).y])
    left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
    right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])
    eye_center = (left_eye + right_eye) / 2
    eye_distance = np.linalg.norm(left_eye - right_eye)
    if eye_distance == 0:
        return 0, "Forward"
    yaw = (nose[0] - eye_center[0]) / eye_distance * 45
    direction = "Left" if yaw < -25 else "Right" if yaw > 25 else "Forward"
    return yaw, direction


# ============================================================
# Session State
# ============================================================
def init_state():
    defaults = {
        'total_frames': 0, 'safe_frames': 0, 'distracted_frames': 0,
        'drowsy_frames': 0, 'alerts': 0, 'class_counts': {},
        'ear_history': [], 'processing': False,
        'current_label': 'Ready', 'current_class': 'idle', 'fps': 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v.copy() if isinstance(v, (dict, list)) else v

init_state()

# Load dlib once
dlib_detector, dlib_predictor = load_dlib()


# ============================================================
# Processing Function
# ============================================================
def process_frame(frame, conf_th=0.4, ear_th=0.20, show_landmarks=False):
    h, w = frame.shape[:2]
    
    # YOLO Detection
    results = yolo_model(frame, verbose=False, conf=conf_th)
    detections = []
    
    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            
            if cls_id in COCO_TO_DRIVER:
                behavior = COCO_TO_DRIVER[cls_id]
                detections.append({
                    'class_name': behavior['class_name'],
                    'color': behavior['color'],
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })
                
                # Track for analytics
                key = behavior['class_name']
                st.session_state.class_counts[key] = st.session_state.class_counts.get(key, 0) + 1
    
    # dlib Face Analysis 
    face_data = {'face_detected': False, 'ear_avg': 1.0, 'yaw_angle': 0,
                 'direction': 'N/A', 'drowsy': False, 'face_rect': None}
    
    if dlib_predictor:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = dlib_detector(gray)
        if len(faces) > 0:
            face = faces[0]
            landmarks = dlib_predictor(gray, face)
            face_data['face_detected'] = True
            face_data['face_rect'] = (face.left(), face.top(), face.right(), face.bottom())
            
            ear = (calculate_ear(landmarks, list(range(36, 42))) + 
                   calculate_ear(landmarks, list(range(42, 48)))) / 2.0
            face_data['ear_avg'] = ear
            
            st.session_state.ear_history.append(ear)
            if len(st.session_state.ear_history) > 30:
                st.session_state.ear_history.pop(0)
            
            smooth = np.mean(st.session_state.ear_history[-15:]) if st.session_state.ear_history else ear
            face_data['drowsy'] = smooth < ear_th
            
            yaw, direction = calculate_head_pose(landmarks)
            face_data['yaw_angle'] = yaw
            face_data['direction'] = direction
            
            if show_landmarks:
                for i in range(68):
                    x, y = landmarks.part(i).x, landmarks.part(i).y
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    
    # Draw detections on frame
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color_bgr = det['color']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1)
    
    if face_data['face_rect']:
        x1, y1, x2, y2 = face_data['face_rect']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
    
    # Status bar overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    
    # Update statistics
    st.session_state.total_frames += 1
    if face_data['drowsy']:
        st.session_state.drowsy_frames += 1
        st.session_state.alerts += 1
        state, sclass = "DROWSY!", "danger"
    elif detections:
        st.session_state.distracted_frames += 1
        st.session_state.alerts += 1
        state, sclass = detections[0]['class_name'], "warning"
    else:
        st.session_state.safe_frames += 1
        state, sclass = "Safe Driving", "safe"
    
    sc = (0, 255, 0) if sclass == "safe" else (0, 255, 255) if sclass == "warning" else (0, 0, 255)
    cv2.putText(frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)
    
    return frame, state, sclass, detections, face_data


# ============================================================
# UI: Title Bar
# ============================================================
status_class = f"status-{st.session_state.current_class}"
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown(f"""
    <div class="title-bar">
        <div>
            <h1>🚗 Driver Behavior Detection Tool</h1>
            <div class="subtitle">YOLO + dlib | Real-Time Driver Monitoring</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_status:
    st.markdown(f"""
    <div style="padding-top: 12px;">
        <div class="status-badge {status_class}">{st.session_state.current_label}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# UI: Controls
# ============================================================
st.markdown('<div class="control-panel">', unsafe_allow_html=True)
col_src, col_model, col_conf, col_ear, col_btn = st.columns([1.2, 1.2, 0.8, 0.8, 1.2])

with col_src:
    source_type = st.selectbox("📹 Source", 
        ["📷 Live Camera", "📁 Upload Video", "🖼️ Upload Image"],
        key="source_type", label_visibility="collapsed")

with col_model:
    model_choice = st.selectbox("🤖 Model",
        ["YOLOv8s (Stable)", "YOLO11s (New)"],
        key="model_choice")

with col_conf:
    conf_th = st.slider("Conf", 0.1, 0.9, 0.6, 0.05, key="conf_th")

with col_ear:
    ear_th = st.slider("EAR", 0.10, 0.30, 0.20, 0.01, key="ear_th")

with col_btn:
    sub1, sub2 = st.columns([1, 1])
    with sub1:
        show_lm = st.checkbox("Landmarks", value=False, key="show_lm")
    with sub2:
        if source_type == "📷 Live Camera":
            if st.button("▶️ Start", type="primary", use_container_width=True, key="start_btn"):
                st.session_state.processing = True
                st.session_state.current_class = 'safe'
                st.session_state.current_label = 'Monitoring...'
                # Reload model in case it changed
                st.cache_resource.clear()
                st.rerun()
        elif st.button("⏹️ Stop", use_container_width=True, key="stop_btn"):
            st.session_state.processing = False
            st.session_state.current_class = 'idle'
            st.session_state.current_label = 'Stopped'

st.markdown('</div>', unsafe_allow_html=True)

# Load YOLO model
yolo_model, CLASS_NAMES, model_path_used = load_yolo_model(model_choice)
# Initialize class counts for new model
for cls_id in CLASS_NAMES:
    if cls_id not in st.session_state.class_counts:
        st.session_state.class_counts[cls_id] = 0

# Show which model is active
st.caption(f"🤖 Active model: **{Path(model_path_used).name}** | Classes: {len(CLASS_NAMES)} | Confidence: {conf_th}")


# ============================================================
# UI: Tabs
# ============================================================
tab1, tab2 = st.tabs(["📹 Monitor", "📊 Analytics"])

with tab1:
    col_vid, col_side = st.columns([2.5, 1])
    
    with col_vid:
        video_placeholder = st.empty()
    
    with col_side:
        total = max(st.session_state.total_frames, 1)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-mini"><div class="value">{st.session_state.total_frames}</div><div class="label">Frames</div></div>
            <div class="metric-mini"><div class="value">{100*st.session_state.safe_frames/total:.0f}%</div><div class="label">Safe</div></div>
            <div class="metric-mini"><div class="value">{100*st.session_state.distracted_frames/total:.0f}%</div><div class="label">Distracted</div></div>
        </div>
        <div class="metric-row" style="margin-top:5px;">
            <div class="metric-mini"><div class="value">{100*st.session_state.drowsy_frames/total:.0f}%</div><div class="label">Drowsy</div></div>
            <div class="metric-mini"><div class="value">{st.session_state.fps:.1f}</div><div class="label">FPS</div></div>
            <div class="metric-mini"><div class="value">{st.session_state.alerts}</div><div class="label">Alerts</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # EAR Gauge
        if st.session_state.ear_history:
            latest_ear = st.session_state.ear_history[-1]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=latest_ear,
                title={'text': "Eye Aspect Ratio (EAR)", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 0.5]},
                    'bar': {'color': "#27AE60" if latest_ear > ear_th else "#E74C3C"},
                    'steps': [
                        {'range': [0, ear_th], 'color': "#FADBD8"},
                        {'range': [ear_th, 0.5], 'color': "#D5F5E3"},
                    ],
                    'threshold': {'line': {'color': "red", 'width': 3}, 'value': ear_th}
                }
            ))
            fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=0))
            st.plotly_chart(fig_gauge, use_container_width=True, key="gauge")

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Behavior Distribution")
        class_data = {CLASS_NAMES[i]: st.session_state.class_counts.get(i, 0) for i in CLASS_NAMES}
        df = pd.DataFrame(list(class_data.items()), columns=['Behavior', 'Count'])
        df = df[df['Count'] > 0]
        if not df.empty:
            colors = ['#E74C3C', '#9B59B6', '#3498DB', '#F39C12', '#2ECC71', '#E67E22', '#1ABC9C', '#E91E63']
            fig1 = px.pie(df, values='Count', names='Behavior', hole=0.4,
                         color_discrete_sequence=colors[:len(df)])
            fig1.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig1, use_container_width=True, key="pie_chart")
        else:
            st.info("No behaviors detected yet.")
    
    with col2:
        st.markdown("### 👁️ EAR History")
        if st.session_state.ear_history:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=list(st.session_state.ear_history), mode='lines',
                                     line=dict(color='#3498DB', width=2), name='EAR'))
            fig2.add_hline(y=ear_th, line_dash="dash", line_color="red", annotation_text="Threshold")
            fig2.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0),
                             xaxis_title="Frame", yaxis_title="EAR", yaxis_range=[0, 0.5])
            st.plotly_chart(fig2, use_container_width=True, key="ear_history_chart")
        else:
            st.info("No EAR data yet.")
    
    # Summary stats
    st.markdown("### 📊 Session Statistics")
    total = max(st.session_state.total_frames, 1)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Frames", st.session_state.total_frames)
    c2.metric("Safe Driving", f"{100*st.session_state.safe_frames/total:.1f}%")
    c3.metric("Distracted", f"{100*st.session_state.distracted_frames/total:.1f}%")
    c4.metric("Drowsy", f"{100*st.session_state.drowsy_frames/total:.1f}%")
    c5.metric("Alerts", st.session_state.alerts)


# ============================================================
# Live Camera Mode
# ============================================================
if source_type == "📷 Live Camera" and st.session_state.processing:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while st.session_state.processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            t0 = time.time()
            processed, state, sclass, dets, fdata = process_frame(
                frame, conf_th, ear_th, show_lm
            )
            st.session_state.fps = 1.0 / max(time.time() - t0, 0.001)
            st.session_state.current_class = sclass
            st.session_state.current_label = state
            
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels="RGB", use_container_width=True)
    
    finally:
        cap.release()
        st.session_state.current_class = 'idle'
        st.session_state.current_label = 'Stopped'


# ============================================================
# Upload Video Mode
# ============================================================
elif source_type == "📁 Upload Video":
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        video_path = tfile.name
        
        st.video(uploaded_file)
        
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            processed_frames = []
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                processed, _, _, _, _ = process_frame(frame, conf_th, ear_th, False)
                processed_frames.append(processed)
                progress_bar.progress((i+1)/total_frames)
                status_text.text(f"Processing... {i+1}/{total_frames}")
            
            cap.release()
            
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            output_path = output_file.name
            output_file.close()
            
            h, w = processed_frames[0].shape[:2]
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
            for f in processed_frames:
                writer.write(f)
            writer.release()
            
            st.success(f"✅ {len(processed_frames)} frames processed!")
            with open(output_path, 'rb') as f:
                st.download_button("📥 Download Processed Video", f,
                                  file_name=f"analyzed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                  mime="video/mp4")
            
            try:
                time.sleep(1)
                if os.path.exists(video_path):
                    os.unlink(video_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except PermissionError:
                pass


# ============================================================
# Upload Image Mode
# ============================================================
elif source_type == "🖼️ Upload Image":
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                processed, state, sclass, dets, fdata = process_frame(
                    frame_bgr, conf_th, ear_th, True
                )
                rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True)
                with col2:
                    st.image(rgb, caption=f"Analysis: {state}", use_container_width=True)
                
                if dets:
                    st.markdown("### 📋 Detected Behaviors")
                    for d in dets:
                        st.markdown(f"- **{d['class_name']}** ({d['confidence']:.1%})")
                
                st.markdown(f"**Face:** {'✅' if fdata['face_detected'] else '❌'} | "
                          f"**EAR:** {fdata['ear_avg']:.3f} | "
                          f"**Head:** {fdata['direction']} | "
                          f"**Drowsy:** {'⚠️ Yes' if fdata['drowsy'] else '✅ No'}")