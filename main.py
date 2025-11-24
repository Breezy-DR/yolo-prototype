import streamlit as st
from PIL import Image
from ultralytics import RTDETR
import cv2
import tempfile
import numpy as np
import os
from inference import get_model
import supervision as sv

st.set_page_config(page_title="Defect Detection", page_icon="🔍", layout="wide")
st.title("Defect Detection App")

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
uploaded_model = st.sidebar.file_uploader("📦 Upload YOLO Model (.pt)", type=["pt"])
conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.25)
input_type = st.sidebar.radio("Select input type", ["Upload Image", "Upload Video", "Camera Stream"])

# --- Save uploaded model to temp file ---
if uploaded_model is not None:
    temp_model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    temp_model_path.write(uploaded_model.read())
    temp_model_path = temp_model_path.name
else:
    st.sidebar.warning("Please upload a YOLO model (.pt) file to continue.")
    st.stop()

# --- Load model ---
@st.cache_resource
def load_model(model_path):
    model = get_model(model_path)
    print("Model loaded successfully.")
    return model

model = load_model(temp_model_path)

# st.markdown("Upload an image/video or use camera stream to detect defects.")

# --- Upload Image ---
if input_type == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)
        if st.button("Start Detection"):
            img = np.array(image)
            pred = model.infer(img)
            result_img = Image.fromarray(results[0].plot())
            with col2:
                st.image(result_img, caption="✅ Detected Image", use_container_width=True)

            boxes = results[0].boxes
            with st.expander("📦 Detected Defects Details"):
                if len(boxes) == 0:
                    st.write("No defects detected.")
                else:
                    for i, box in enumerate(boxes, 1):
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = [round(x,2) for x in box.xyxy[0].tolist()]
                        defect_name = model.names[cls]
                        st.write(f"**{i}. Defect:** {defect_name} | **Confidence:** {conf:.2f}")
                        st.caption(f"Bounding Box (xyxy): {xyxy}")

# --- Upload Video (stop on first detection) ---
elif input_type == "Upload Video":
    uploaded_file = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        if st.button("Start Video Detection"):
            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()
            st.subheader("✅ First Frame with Defect")
            captured_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model.predict(source=img_rgb, conf=conf_threshold, imgsz=640)
                boxes = results[0].boxes

                if len(boxes) > 0:
                    captured_count += 1
                    result_img = Image.fromarray(results[0].plot())
                    st.image(result_img, caption=f"Frame #{captured_count} with defects", use_container_width=True)
                    with st.expander("📦 Details"):
                        for i, box in enumerate(boxes, 1):
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            xyxy = [round(x,2) for x in box.xyxy[0].tolist()]
                            defect_name = model.names[cls]
                            st.write(f"**{i}. Defect:** {defect_name} | **Confidence:** {conf:.2f}")
                            st.caption(f"Bounding Box (xyxy): {xyxy}")
                    break  # stop video on first detection
            cap.release()

# --- Camera Stream (stop on first detection) ---
elif input_type == "Camera Stream":
    if st.button("Start Camera Stream"):
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open camera")
            st.stop()
        st.write("🔴 Streaming started. Will stop after first detected defect.")
        captured_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(source=img_rgb, conf=conf_threshold, imgsz=640)
            boxes = results[0].boxes

            if len(boxes) > 0:
                captured_count += 1
                result_img = Image.fromarray(results[0].plot())
                st.subheader(f"✅ Defect Detected! Captured Frame #{captured_count}")
                st.image(result_img, use_container_width=True)
                with st.expander("📦 Details"):
                    for i, box in enumerate(boxes, 1):
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = [round(x,2) for x in box.xyxy[0].tolist()]
                        defect_name = model.names[cls]
                        st.write(f"**{i}. Defect:** {defect_name} | **Confidence:** {conf:.2f}")
                        st.caption(f"Bounding Box (xyxy): {xyxy}")
                break  # stop streaming on first detection

            stframe.image(img_rgb, channels="RGB")
        cap.release()
        st.write("🔴 Streaming stopped.")
