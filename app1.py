import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="YOLOv11 Leaf Disease Detection",
    page_icon="🌱",
    layout="centered"
)

st.title("🌿 YOLOv11-Based Leaf Disease Detection")
st.markdown(
    "**Research Prototype – Smart Farming | Model Evaluation & Batch Testing**"
)

# ---------------- MODEL PATHS ----------------
MODEL_PATHS = {
    "YOLOv11n (Nano)": "model/best_yolo11n.pt",
    "YOLOv11s (Small)": "model/best_yolo11s.pt",
    "YOLOv11m (Medium)": "model/best_yolo11m.pt"
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
        "Select YOLOv11 Model",
        list(MODEL_PATHS.keys()),
        key="single_model"
    )

    model = load_model(MODEL_PATHS[model_choice])

    if st.button("Run Detection"):
        start = time.time()
        results = model.predict(np.array(image), save=False)
        end = time.time()

        annotated = results[0].plot()
        st.image(annotated, caption="Detection Output", width="stretch")

        st.subheader("Detection Confidence Scores")
        for i, box in enumerate(results[0].boxes):
            cls = model.names[int(box.cls)]
            conf = float(box.conf)
            st.write(f"Detection {i+1}: {cls} — {conf:.3f}")

        st.info(f"Inference Time: {(end - start)*1000:.2f} ms")

# =====================================================
# SECTION 2: BATCH TESTING FOR EXPERIMENTS
# =====================================================
st.divider()
st.header("🧪 Batch Testing & Model Comparison")

batch_files = st.file_uploader(
    "Upload multiple leaf images (Batch Testing)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if batch_files:
    st.success(f"{len(batch_files)} images loaded for batch testing")

    run_batch = st.button("Run Batch Evaluation on All Models")

    if run_batch:
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
                "Total Detections": total_detections,
                "Average Confidence": np.mean(confidences) if confidences else 0,
                "Avg Inference Time (ms)": np.mean(inference_times) * 1000
            })

        df = pd.DataFrame(metrics)

        # ---------------- TABLE ----------------
        st.subheader("📊 Model Comparison Metrics")
        st.dataframe(df, use_container_width=True)

        # ---------------- VISUALIZATION ----------------
        st.subheader("📈 Metrics Visualization")

        fig, ax = plt.subplots()
        ax.bar(df["Model"], df["Avg Inference Time (ms)"])
        ax.set_ylabel("Inference Time (ms)")
        ax.set_title("Average Inference Time per Model")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.bar(df["Model"], df["Average Confidence"])
        ax.set_ylabel("Average Confidence")
        ax.set_title("Average Detection Confidence per Model")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.bar(df["Model"], df["Total Detections"])
        ax.set_ylabel("Total Detections")
        ax.set_title("Total Detections Across Batch")
        st.pyplot(fig)

        st.success("Batch evaluation completed successfully")

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "YOLOv11 Multi-Model Evaluation for Smart Farming Applications"
)
