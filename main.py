import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="YOLOv8 Defect Detection", page_icon="🔍", layout="centered")

st.title("🔍 YOLOv8 Defect Detection App")
st.markdown("Upload an image to detect defects using your trained YOLO model.")

# --- Load YOLO model ---
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model(MODEL_PATH)

# --- Upload Image ---
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # --- Run YOLO prediction ---
    st.write("🔍 Running YOLO detection...")
    with st.spinner("Detecting..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            conf_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.25)
            results = model.predict(source=tmp.name, imgsz=640, conf=conf_threshold)

    # --- Display results ---
    st.subheader("✅ Detection Results")

    # Display annotated image
    result_image = Image.fromarray(results[0].plot())
    st.image(result_image, caption="Detected Image", use_container_width=True)

    # --- Show detected classes & confidence ---
    boxes = results[0].boxes
    if len(boxes) == 0:
        st.warning("No defects detected.")
    else:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [round(x, 2) for x in box.xyxy[0].tolist()]
            defect_name = model.names[cls]
            st.write(f"**Defect:** {defect_name} | **Confidence:** {conf:.2f}")
            st.caption(f"Bounding Box (xyxy): {xyxy}")