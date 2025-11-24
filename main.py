import streamlit as st
from PIL import Image
import cv2
import tempfile
import numpy as np
import os
import gc
import torch
import pandas as pd
from rfdetr import RFDETRMedium

APP_TITLE = "Defect Detection App"
MODEL_FILENAME = "uploaded_model.pt"
BOX_COLOR = "#FF0000"
hex_color = BOX_COLOR.lstrip('#')
r = int(hex_color[0:2], 16)
g = int(hex_color[2:4], 16)
b = int(hex_color[4:6], 16)
rgb_color = (r, g, b)
BOX_THICKNESS = 6
DEFAULT_CONF = 0.25

st.set_page_config(page_title=APP_TITLE, page_icon="🔍", layout="wide")
st.title(APP_TITLE)

st.sidebar.header("⚙️ Settings")
uploaded_model = st.sidebar.file_uploader("📦 Upload RF-DETR Model (.pt)", type=["pt"])
conf_threshold = st.sidebar.slider("Confidence threshold", 0.01, 1.0, float(DEFAULT_CONF))
input_type = st.sidebar.radio("Select input type", ["Upload Image"])

if uploaded_model is not None:
    with open(MODEL_FILENAME, "wb") as f:
        f.write(uploaded_model.read())
elif not os.path.exists(MODEL_FILENAME):
    st.sidebar.warning("Please upload a RF-DETR model (.pt) file to continue.")
    st.stop()

@st.cache_resource
def load_model(model_path: str):
    model = RFDETRMedium(pretrain_weights=model_path)
    class_names = getattr(model, "class_names", None) or getattr(model, "names", None)
    return {"model": model, "class_names": class_names}

_model_bundle = load_model(MODEL_FILENAME)
model = _model_bundle["model"]
class_names = _model_bundle["class_names"] or []

st.sidebar.success("✅ Model loaded and cached.")

def predict_image(np_image: np.ndarray, threshold: float):
    return model.predict(np_image, threshold=threshold)

def annotate_cv2(image: np.ndarray, detections, class_names):
    annotated = image.copy()
    for cls_id, conf, box in zip(detections.class_id, detections.confidence, detections.xyxy):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        label = f"{class_names[int(cls_id)]} {float(conf):.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), rgb_color, thickness=BOX_THICKNESS)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(annotated, (x1, y1 - text_height - 10), (x1 + text_width, y1), rgb_color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return annotated

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if input_type == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(pil_image)
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_image, caption="🖼️ Uploaded Image", use_container_width=True)
        if st.button("Start Detection"):
            with st.spinner("Running detection..."):
                detections = predict_image(img_np, threshold=conf_threshold)
                annotated = annotate_cv2(img_np, detections, class_names)
                with col2:
                    st.image(annotated, caption="✅ Detected Image", use_container_width=True)

                with st.expander("📦 Detected Defects Details", expanded=False):
                    if len(detections) == 0:
                        st.write("No defects detected.")
                    else:
                        df = pd.DataFrame({
                            "Class": [class_names[int(c)] for c in detections.class_id],
                            "Confidence": [round(float(x), 3) for x in detections.confidence],
                            "X1": [round(float(b[0]), 2) for b in detections.xyxy],
                            "Y1": [round(float(b[1]), 2) for b in detections.xyxy],
                            "X2": [round(float(b[2]), 2) for b in detections.xyxy],
                            "Y2": [round(float(b[3]), 2) for b in detections.xyxy],
                        })
                        st.dataframe(df, use_container_width=True)
                cleanup()
# # ---------------------
# # UI: Upload Video (stop on first detection)
# # ---------------------
# elif input_type == "Upload Video":
#     uploaded_file = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov"])
#     if uploaded_file:
#         tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#         tfile.write(uploaded_file.read())
#         video_path = tfile.name
#
#         if st.button("Start Video Detection"):
#             cap = cv2.VideoCapture(video_path)
#             stframe = st.empty()
#             st.subheader("✅ First Frame with Defect")
#             frame_count = 0
#             try:
#                 while cap.isOpened():
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#                     frame_count += 1
#
#                     # reduce frame rate/size for faster processing
#                     small = cv2.resize(frame, (640, 640))
#                     img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
#
#                     detections = predict_image(img_rgb, threshold=conf_threshold)
#
#                     if len(detections) > 0:
#                         labels = [
#                             f"{(class_names[int(c)] if len(class_names) > int(c) else int(c))} {float(conf):.2f}"
#                             for c, conf in zip(detections.class_id, detections.confidence)
#                         ]
#
#                         annotated = box_annotator.annotate(img_rgb.copy(), detections)
#                         annotated = label_annotator.annotate(annotated, detections, labels)
#
#                         st.image(annotated, caption=f"Frame #{frame_count} with defects", use_container_width=True)
#
#                         with st.expander("📦 Details"):
#                             for i, (c, conf, box) in enumerate(zip(detections.class_id, detections.confidence, detections.xyxy), 1):
#                                 cls_name = class_names[int(c)] if len(class_names) > int(c) else int(c)
#                                 st.write(f"**{i}. Defect:** {cls_name} | **Confidence:** {float(conf):.2f}")
#                                 st.caption(f"Bounding Box (xyxy): {[round(float(x),2) for x in box]}")
#
#                         break  # stop video on first detection
#
#                     # show live frame while searching
#                     stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
#
#             finally:
#                 cap.release()
#                 cleanup()
#
# # ---------------------
# # UI: Camera Stream (stop on first detection)
# # ---------------------
# elif input_type == "Camera Stream":
#     if st.button("Start Camera Stream"):
#         stframe = st.empty()
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             st.error("Cannot open camera")
#             st.stop()
#
#         st.write("🔴 Streaming started. Will stop after first detected defect.")
#         frame_count = 0
#         try:
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 frame_count += 1
#
#                 # resize to speed up
#                 small = cv2.resize(frame, (640, 640))
#                 img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
#
#                 detections = predict_image(img_rgb, threshold=conf_threshold)
#
#                 if len(detections) > 0:
#                     labels = [
#                         f"{(class_names[int(c)] if len(class_names) > int(c) else int(c))} {float(conf):.2f}"
#                         for c, conf in zip(detections.class_id, detections.confidence)
#                     ]
#
#                     annotated = box_annotator.annotate(img_rgb.copy(), detections)
#                     annotated = label_annotator.annotate(annotated, detections, labels)
#
#                     st.subheader(f"✅ Defect Detected! Captured Frame #{frame_count}")
#                     st.image(annotated, use_container_width=True)
#
#                     with st.expander("📦 Details"):
#                         for i, (c, conf, box) in enumerate(zip(detections.class_id, detections.confidence, detections.xyxy), 1):
#                             cls_name = class_names[int(c)] if len(class_names) > int(c) else int(c)
#                             st.write(f"**{i}. Defect:** {cls_name} | **Confidence:** {float(conf):.2f}")
#                             st.caption(f"Bounding Box (xyxy): {[round(float(x),2) for x in box]}")
#
#                     break  # stop streaming on first detection
#
#                 stframe.image(img_rgb, channels="RGB")
#
#         finally:
#             cap.release()
#             cleanup()

# ---------------------
# Notes
# ---------------------
st.markdown("---")
st.caption("Notes:\n- The model is saved to `uploaded_model.pt` in the app directory so the model loads once and is cached.\n- Bounding boxes are forced to be red and thick (see BOX_COLOR and BOX_THICKNESS).\n- The app performs aggressive memory cleanup after each detection to avoid running out of resources.")
