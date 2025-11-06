import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile
import time
import numpy as np
import base64
import os

st.set_page_config(page_title="Defect Detection", page_icon="🔍", layout="wide")
st.title("Defect Detection App")

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
MODEL_PATH = st.sidebar.text_input("Model path", "short shot_case blower jk.pt")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.25)
input_type = st.sidebar.radio("Select input type", ["Upload Image", "Upload Video", "Camera Stream"])

DEFAULT_FPS = 25
MAX_RUNTIME_SEC = 50
FRAME_FAIL_SLEEP = 0.1
DEFECT_CLASSES = {"scratch", "crack", "bubble", "silver", "short_shoot"}

# --- Load model ---
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

if not os.path.exists(MODEL_PATH):
    st.sidebar.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model(MODEL_PATH)

def run_uploaded_video_detection(camera_id, uploaded_file, model, conf_threshold=0.4):

    # Save uploaded file to temp location so OpenCV can read it
    temp_video_path = "temp_uploaded_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        st.error("❌ Cannot open uploaded video.")
        return

    stframe = st.empty()
    st.subheader("🎥 Processing Uploaded Video...")

    fps = DEFAULT_FPS
    start_time = time.time()
    frame_index = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed > MAX_RUNTIME_SEC:
            st.info(f"⌛ Timeout reached ({MAX_RUNTIME_SEC}s), no defect found.")
            break

        ret, frame = cap.read()
        if not ret:
            time.sleep(FRAME_FAIL_SLEEP)
            continue

        frame_index += 1
        time.sleep(1 / fps)

        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(display_frame, channels="RGB", use_container_width=True)

        results = model.predict(source=frame, conf=conf_threshold, imgsz=640, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            st.subheader("✅ Defect Detected")
            result_img = Image.fromarray(results[0].plot())
            st.image(result_img, use_container_width=True)

            for i, box in enumerate(boxes, 1):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = [round(x, 2) for x in box.xyxy[0].tolist()]
                defect_name = model.names.get(cls, f"class_{cls}")

                if defect_name.lower() not in DEFECT_CLASSES:
                    continue

                st.write(f"**{i}. Defect:** `{defect_name}`  |  **Confidence:** `{conf:.2f}`")
                st.caption(f"Bounding Box (xyxy): `{xyxy}`")

                timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                save_dir = "./outputs/images"
                os.makedirs(save_dir, exist_ok=True)
                image_filename = f"{camera_id}_{defect_name}_{timestamp}.jpg"
                image_filepath = os.path.join(save_dir, image_filename)
                cv2.imwrite(image_filepath, frame)
                st.success(f"🖼️ Saved defect frame → `{image_filename}`")

                _, buffer = cv2.imencode(".jpg", frame)
                frame_base64 = base64.b64encode(buffer).decode("utf-8")

            break  # stop after first detection

    cap.release()

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
            results = model.predict(source=np.array(image), conf=conf_threshold, imgsz=640)
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
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video and st.button("Start Detection"):
        run_uploaded_video_detection(
            camera_id="UPLOADED_CAM",
            uploaded_file=uploaded_video,
            model=model,
            conf_threshold=0.4
        )

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
