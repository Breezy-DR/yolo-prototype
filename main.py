import streamlit as st
from PIL import Image
import cv2
import tempfile
import numpy as np
import os
import gc
import torch
import pandas as pd
import supervision as sv
from rfdetr import RFDETRMedium

# ---------------------
# Config
# ---------------------
APP_TITLE = "Defect Detection App"
MODEL_FILENAME = "uploaded_model.pt"  # stable filename to keep cache consistent
BOX_COLOR = sv.Color(r=255, g=255, b=0).as_rgb() # (255, 255, 0)  # red in RGB
BOX_THICKNESS = 4  # pretty thick boxes (keeps original appearance)
DEFAULT_CONF = 0.25

st.set_page_config(page_title=APP_TITLE, page_icon="🔍", layout="wide")
st.title(APP_TITLE)

# ---------------------
# Sidebar / Settings
# ---------------------
st.sidebar.header("⚙️ Settings")
uploaded_model = st.sidebar.file_uploader("📦 Upload RF-DETR Model (.pt)", type=["pt"])
conf_threshold = st.sidebar.slider("Confidence threshold", 0.01, 1.0, float(DEFAULT_CONF))
input_type = st.sidebar.radio("Select input type", ["Upload Image", "Upload Video", "Camera Stream"])

# Save uploaded model to a stable path so caching works reliably
if uploaded_model is not None:
    with open(MODEL_FILENAME, "wb") as f:
        f.write(uploaded_model.read())
elif not os.path.exists(MODEL_FILENAME):
    st.sidebar.warning("Please upload a RF-DETR model (.pt) file to continue.")
    st.stop()

# ---------------------
# Load model (cached)
# ---------------------
@st.cache_resource
def load_model(model_path: str):
    # Load RFDETR model once and cache it across reruns
    model = RFDETRMedium(pretrain_weights=model_path)

    # Some RFDETR versions expose `class_names`, others `names`
    class_names = getattr(model, "class_names", None) or getattr(model, "names", None)

    # simple wrapper that keeps the model instance and names handy
    return {
        "model": model,
        "class_names": class_names
    }

_model_bundle = load_model(MODEL_FILENAME)
model = _model_bundle["model"]
class_names = _model_bundle["class_names"] or []

st.sidebar.success("✅ Model loaded and cached.")

# ---------------------
# Helper utilities
# ---------------------
box_annotator = sv.BoxAnnotator(thickness=BOX_THICKNESS, color=BOX_COLOR)
label_annotator = sv.LabelAnnotator()


def predict_image(np_image: np.ndarray, threshold: float):
    """Run model.predict and return detections in supervision Detections form.
    This isolates torch tensors and helps to cleanup memory after use."""
    # RF-DETR model might accept PIL or numpy; we send numpy for lighter memory usage
    result = model.predict(np_image, threshold=threshold)
    return result


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------------------
# UI: Upload Image
# ---------------------
if input_type == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.image(pil_image, caption="🖼️ Uploaded Image", use_container_width=True)

        run_detection = st.button("Start Detection")

        if run_detection:
            with st.spinner("Running detection..."):
                # convert to numpy to avoid keeping PIL in session state
                img_np = np.array(pil_image)

                detections = predict_image(img_np, threshold=conf_threshold)

                # Labels using model's class names (fallback to indices if missing)
                labels = [
                    f"{(class_names[int(c)] if len(class_names) > int(c) else int(c))} {float(conf):.2f}"
                    for c, conf in zip(detections.class_id, detections.confidence)
                ]

                # annotate (keep color and thickness as specified)
                annotated = box_annotator.annotate(img_np.copy(), detections)
                annotated = label_annotator.annotate(annotated, detections, labels)

                with col2:
                    st.image(annotated, caption="✅ Detected Image", use_container_width=True)

                # Details table (adjustable)
                with st.expander("📦 Detected Defects Details", expanded=False):
                    if len(detections) == 0:
                        st.write("No defects detected.")
                    else:
                        df = pd.DataFrame({
                            "Class": [class_names[int(c)] if len(class_names) > int(c) else int(c) for c in detections.class_id],
                            "Confidence": [round(float(x), 3) for x in detections.confidence],
                            "X1": [round(float(b[0]), 2) for b in detections.xyxy],
                            "Y1": [round(float(b[1]), 2) for b in detections.xyxy],
                            "X2": [round(float(b[2]), 2) for b in detections.xyxy],
                            "Y2": [round(float(b[3]), 2) for b in detections.xyxy],
                        })
                        st.dataframe(df, use_container_width=True)

                # cleanup to prevent memory growth across multiple uploads
                cleanup()

# ---------------------
# UI: Upload Video (stop on first detection)
# ---------------------
elif input_type == "Upload Video":
    uploaded_file = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        if st.button("Start Video Detection"):
            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()
            st.subheader("✅ First Frame with Defect")
            frame_count = 0
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1

                    # reduce frame rate/size for faster processing
                    small = cv2.resize(frame, (640, 640))
                    img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                    detections = predict_image(img_rgb, threshold=conf_threshold)

                    if len(detections) > 0:
                        labels = [
                            f"{(class_names[int(c)] if len(class_names) > int(c) else int(c))} {float(conf):.2f}"
                            for c, conf in zip(detections.class_id, detections.confidence)
                        ]

                        annotated = box_annotator.annotate(img_rgb.copy(), detections)
                        annotated = label_annotator.annotate(annotated, detections, labels)

                        st.image(annotated, caption=f"Frame #{frame_count} with defects", use_container_width=True)

                        with st.expander("📦 Details"):
                            for i, (c, conf, box) in enumerate(zip(detections.class_id, detections.confidence, detections.xyxy), 1):
                                cls_name = class_names[int(c)] if len(class_names) > int(c) else int(c)
                                st.write(f"**{i}. Defect:** {cls_name} | **Confidence:** {float(conf):.2f}")
                                st.caption(f"Bounding Box (xyxy): {[round(float(x),2) for x in box]}")

                        break  # stop video on first detection

                    # show live frame while searching
                    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            finally:
                cap.release()
                cleanup()

# ---------------------
# UI: Camera Stream (stop on first detection)
# ---------------------
elif input_type == "Camera Stream":
    if st.button("Start Camera Stream"):
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open camera")
            st.stop()

        st.write("🔴 Streaming started. Will stop after first detected defect.")
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                # resize to speed up
                small = cv2.resize(frame, (640, 640))
                img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                detections = predict_image(img_rgb, threshold=conf_threshold)

                if len(detections) > 0:
                    labels = [
                        f"{(class_names[int(c)] if len(class_names) > int(c) else int(c))} {float(conf):.2f}"
                        for c, conf in zip(detections.class_id, detections.confidence)
                    ]

                    annotated = box_annotator.annotate(img_rgb.copy(), detections)
                    annotated = label_annotator.annotate(annotated, detections, labels)

                    st.subheader(f"✅ Defect Detected! Captured Frame #{frame_count}")
                    st.image(annotated, use_container_width=True)

                    with st.expander("📦 Details"):
                        for i, (c, conf, box) in enumerate(zip(detections.class_id, detections.confidence, detections.xyxy), 1):
                            cls_name = class_names[int(c)] if len(class_names) > int(c) else int(c)
                            st.write(f"**{i}. Defect:** {cls_name} | **Confidence:** {float(conf):.2f}")
                            st.caption(f"Bounding Box (xyxy): {[round(float(x),2) for x in box]}")

                    break  # stop streaming on first detection

                stframe.image(img_rgb, channels="RGB")

        finally:
            cap.release()
            cleanup()

# ---------------------
# Notes
# ---------------------
st.markdown("---")
st.caption("Notes:\n- The model is saved to `uploaded_model.pt` in the app directory so the model loads once and is cached.\n- Bounding boxes are forced to be red and thick (see BOX_COLOR and BOX_THICKNESS).\n- The app performs aggressive memory cleanup after each detection to avoid running out of resources.")
