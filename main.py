import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os
import gc
import torch

import supervision as sv
from rfdetr import RFDETRMedium


# ---------------------------------------------------------
# Streamlit App Config
# ---------------------------------------------------------
st.set_page_config(page_title="Defect Detection", page_icon="🔍", layout="wide")
st.title("RF-DETR Defect Detection")


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("⚙️ Settings")

uploaded_model = st.sidebar.file_uploader(
    "📦 Upload RF-DETR Model (.pt)",
    type=["pt"]
)

conf_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25
)

input_type = st.sidebar.radio(
    "Select input type",
    ["Upload Image"],
    captions=["Detect defects from an image file"]
)


# ---------------------------------------------------------
# Save Uploaded Model Persistently (Stable Path)
# ---------------------------------------------------------
MODEL_PATH = "uploaded_model.pt"

if uploaded_model is not None:
    with open(MODEL_PATH, "wb") as f:
        f.write(uploaded_model.read())
else:
    st.sidebar.warning("Please upload an RF-DETR model (.pt) file to continue.")
    st.stop()


# ---------------------------------------------------------
# Load Model (Only Once)
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    model = RFDETRMedium(pretrain_weights=model_path)
    return model


model = load_model(MODEL_PATH)


# ---------------------------------------------------------
# Image Processor Layout
# ---------------------------------------------------------
if input_type == "Upload Image":

    uploaded_file = st.file_uploader(
        "📤 Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        # Display original image
        with col1:
            st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        run_detection = st.button("Start Detection")

        if run_detection:

            # Convert PIL → numpy (smaller memory)
            image_np = np.array(image)

            # Run model
            detections_raw = model.predict(image_np, threshold=conf_threshold)

            detections = sv.Detections.from_inference(detections_raw)

            # Prepare labels
            labels = [
                f"{model.class_names[int(cls)]} {conf:.2f}"
                for cls, conf in zip(detections.class_id, detections.confidence)
            ]

            box_annotator = sv.BoxAnnotator(
                thickness=4,
                color=sv.Color.from_hex("#FF0000")  # FORCE RED
            )

            label_annotator = sv.LabelAnnotator(
                text_thickness=2,
                text_scale=0.7,
            )

            annotated = box_annotator.annotate(
                scene=image_np.copy(),
                detections=detections
            )

            annotated = label_annotator.annotate(
                scene=annotated,
                detections=detections,
                labels=labels
            )

            # Show results
            with col2:
                st.image(annotated, caption="✅ Detection Result", use_container_width=True)

            # ---------------------------------------------------------
            # Detection Details Table
            # ---------------------------------------------------------
            with st.expander("📦 Detected Defects Details"):

                if len(detections) == 0:
                    st.write("No defects detected.")
                else:
                    import pandas as pd

                    df = pd.DataFrame({
                        "Class": [model.class_names[int(c)] for c in detections.class_id],
                        "Confidence": [round(float(x), 3) for x in detections.confidence],
                        "X1": [round(float(b[0]), 2) for b in detections.xyxy],
                        "Y1": [round(float(b[1]), 2) for b in detections.xyxy],
                        "X2": [round(float(b[2]), 2) for b in detections.xyxy],
                        "Y2": [round(float(b[3]), 2) for b in detections.xyxy],
                    })

                    st.dataframe(df, use_container_width=True)

            # ---------------------------------------------------------
            # Memory Cleanup
            # ---------------------------------------------------------
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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
