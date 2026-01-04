import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="YOLOv11 Leaf Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# -------------------- HEADER --------------------
st.title("🌿 YOLOv11-Based Leaf Disease Detection")
st.markdown(
    """
    *Transforming Smart Farming using Deep Learning*
    """
)

st.divider()

# -------------------- IMAGE UPLOAD --------------------
uploaded_file = st.file_uploader(
    "Upload a vegetable leaf image",
    type=["jpg", "jpeg", "png"]
)

# -------------------- AFTER IMAGE UPLOAD --------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(
        image,
        caption="Uploaded Leaf Image",
        width="stretch"
    )

    st.divider()

    # -------------------- MODEL SELECTION --------------------
    st.subheader("Select YOLOv11 Model for Inference")

    model_option = st.selectbox(
        "Choose model",
        (
            "YOLOv11n (Nano – Lightweight)",
            "YOLOv11s (Small – Balanced)",
            "YOLOv11m (Medium – Higher Accuracy)"
        )
    )

    model_paths = {
        "YOLOv11n (Nano – Lightweight)": "model/best_yolo11n.pt",
        "YOLOv11s (Small – Balanced)": "model/best_yolo11s.pt",
        "YOLOv11m (Medium – Higher Accuracy)": "model/best_yolo11m.pt"
    }

    # -------------------- LOAD MODEL --------------------
    @st.cache_resource
    def load_model(model_path):
        return YOLO(model_path)

    model = load_model(model_paths[model_option])
    
    # -------------------- DETECTION BUTTON --------------------
    if st.button("🔍 Run Disease Detection"):
        with st.spinner("Running YOLOv11 inference..."):
            results = model.predict(
                source=np.array(image),
                save=False
            )

        # -------------------- DISPLAY OUTPUT --------------------
        annotated_image = results[0].plot()
        st.image(
            annotated_image,
            caption="Detection Output",
            width="stretch"
        )

        st.divider()

        # -------------------- CONFIDENCE DISPLAY --------------------
        st.subheader("Detection Confidence Scores")

        if len(results[0].boxes) == 0:
            st.warning("No diseases detected in the uploaded image.")
        else:
            for i, box in enumerate(results[0].boxes):
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[class_id]

                st.write(
                    f"**Detection {i+1}:** "
                    f"{class_name} — Confidence: **{confidence:.3f}**"
                )

# -------------------- FOOTER --------------------
st.divider()
st.caption(
    "YOLOv11-based Leaf Disease Detection System"
)
