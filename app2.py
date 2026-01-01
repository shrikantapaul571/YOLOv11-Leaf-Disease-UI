import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Leaf Disease Detection (YOLOv8 vs YOLOv11)",
    page_icon="🌱",
    layout="centered"
)

st.title("🌿 Leaf Disease Detection using YOLOv8 and YOLOv11")
st.markdown(
    "**Smart Farming Research Prototype – Cross-Model Evaluation**"
)

# ---------------- MODEL PATHS ----------------
MODEL_PATHS = {
    # YOLOv11
    "YOLOv11n (Nano)": "model/yolo11/best_yolo11n.pt",
    "YOLOv11s (Small)": "model/yolo11/best_yolo11s.pt",
    "YOLOv11m (Medium)": "model/yolo11/best_yolo11m.pt",

    # YOLOv8
    "YOLOv8n (Nano)": "model/yolo8/best_yolo8n.pt",
    "YOLOv8s (Small)": "model/yolo8/best_yolo8s.pt",
    "YOLOv8m (Medium)": "model/yolo8/best_yolo8m.pt",
}

@st.cache_resource
def load_model(path):
    return YOLO(path)

st.divider()

# =====================================================
# SECTION 1: SINGLE IMAGE INFERENCE
# =====================================================
st.header("🔍 Single Image Inference")

uploaded_file = st.file_uploader(
    "Upload a vegetable leaf image",
    type=["jpg", "jpeg", "png"],
    key="single"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    model_choice = st.selectbox(
        "Select Model",
        list(MODEL_PATHS.keys())
    )

    model = load_model(MODEL_PATHS[model_choice])

    if st.button("Run Detection"):
        start = time.time()
        results = model.predict(np.array(image), save=False)
        end = time.time()

        annotated = results[0].plot()
        st.image(annotated, caption="Detection Output", width="stretch")

        st.subheader("Detection Confidence Scores")
        if len(results[0].boxes) == 0:
            st.warning("No diseases detected.")
        else:
            for i, box in enumerate(results[0].boxes):
                cls = model.names[int(box.cls)]
                conf = float(box.conf)
                st.write(f"Detection {i+1}: {cls} — {conf:.3f}")

        st.info(f"Inference Time: {(end - start)*1000:.2f} ms")

# =====================================================
# SECTION 2: BATCH TESTING & MODEL COMPARISON
# =====================================================
st.divider()
st.header("🧪 Batch Testing & Model Comparison")

batch_files = st.file_uploader(
    "Upload multiple leaf images (Batch Evaluation)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if batch_files:
    st.success(f"{len(batch_files)} images loaded")

    if st.button("Run Batch Evaluation on All Models"):
        metrics = []

        for model_name, model_path in MODEL_PATHS.items():
            model = load_model(model_path)

            total_detections = 0
            confidences = []
            inference_times = []

            for file in batch_files:
                img = Image.open(file).convert("RGB")
                img_np = np.array(img)

                start = time.time()
                results = model.predict(img_np, save=False)
                end = time.time()

                inference_times.append(end - start)

                for box in results[0].boxes:
                    total_detections += 1
                    confidences.append(float(box.conf))

            metrics.append({
                "Model": model_name,
                "Architecture": "YOLOv11" if "11" in model_name else "YOLOv8",
                "Total Detections": total_detections,
                "Average Confidence": np.mean(confidences) if confidences else 0,
                "Avg Inference Time (ms)": np.mean(inference_times) * 1000
            })

        df = pd.DataFrame(metrics)

        # ---------------- TABLE ----------------
        st.subheader("📊 Model Comparison Table")
        st.dataframe(df, use_container_width=True)

        # ---------------- VISUALIZATION ----------------
        st.subheader("📈 Metrics Visualization")

        for metric, ylabel in [
            ("Avg Inference Time (ms)", "Inference Time (ms)"),
            ("Average Confidence", "Average Confidence"),
            ("Total Detections", "Total Detections")
        ]:
            fig, ax = plt.subplots()
            ax.bar(df["Model"], df[metric])
            ax.set_ylabel(ylabel)
            ax.set_title(metric)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

        st.success("Batch evaluation completed successfully")

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Cross-Generation YOLOv8 vs YOLOv11 Evaluation for Smart Farming"
)
