
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
from pathlib import Path
import time

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔬",
    layout="wide"
)

# =====================================================
# PROJECT CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pcb_defect_best.pt"

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640

MODEL_MAP50 = 99.25

CLASS_NAMES = {
    0: "Mouse Bite",
    1: "Spur",
    2: "Missing Hole",
    3: "Short",
    4: "Open Circuit",
    5: "Spurious Copper"
}

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 36px;
    font-weight: 750;
    color: #2471A3;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 16px;
    color: #9CA3AF;
    margin-bottom: 20px;
}

.metric-card {
    background-color: #172B3A;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #28465C;
    text-align: center;
}

.metric-label {
    color: #B8C7D1;
    font-size: 15px;
}

.metric-value {
    color: #55D6A3;
    font-size: 32px;
    font-weight: 700;
    margin-top: 8px;
}

.metric-caption {
    color: #9CA3AF;
    font-size: 12px;
    margin-top: 5px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("PCB Inspection")

st.sidebar.markdown("### Model Information")

st.sidebar.write("**Architecture:** YOLOv8s")
st.sidebar.write("**Input Size:** 640 × 640")
st.sidebar.write("**Defect Classes:** 6")


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="main-title">PCB Defect Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Printed Circuit Board inspection using YOLOv8'
    '</div>',
    unsafe_allow_html=True
)


# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.markdown("### Model Performance")

st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">mAP@0.5</div>
    <div class="metric-value">{MODEL_MAP50:.2f}%</div>
    <div class="metric-caption">
        Evaluated on 1,063 held-out test images
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# =====================================================
# IMAGE UPLOAD
# =====================================================

st.subheader("Upload PCB Image")

uploaded_file = st.file_uploader(
    "Choose a PCB image for inspection",
    type=["jpg", "jpeg", "png"]
)


# =====================================================
# IMAGE PROCESSING
# =====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("### Image Inspection")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button(
        "Detect Defects",
        type="primary",
        use_container_width=True
    ):

        try:

            model = load_model()

            with st.spinner("Analysing PCB image..."):

                start_time = time.time()

                results = model.predict(
                    source=image,
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=IMAGE_SIZE,
                    verbose=False
                )

                inference_time = time.time() - start_time

            result = results[0]

            annotated_image = result.plot()
            annotated_image = annotated_image[:, :, ::-1]

            with col2:
                st.subheader("Detection Results")
                st.image(
                    annotated_image,
                    use_container_width=True
                )

            # =================================================
            # EXTRACT DETECTIONS
            # =================================================

            defect_data = []

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())

                    defect_name = CLASS_NAMES.get(
                        class_id,
                        f"Class {class_id}"
                    )

                    defect_data.append({
                        "Defect": defect_name,
                        "Confidence (%)": round(
                            confidence * 100, 2
                        )
                    })

            # =================================================
            # DETECTION SUMMARY
            # =================================================

            st.divider()
            st.subheader("Detection Summary")

            total_defects = len(defect_data)

            unique_defects = len(
                set(item["Defect"] for item in defect_data)
            )

            c1, c2, c3 = st.columns(3)

            c1.metric("Total Defects", total_defects)
            c2.metric(
                "Inference Time",
                f"{inference_time:.3f} sec"
            )
            c3.metric("Defect Types", unique_defects)

            # =================================================
            # DETECTION RESULTS TABLE
            # =================================================

            if defect_data:

                st.success(
                    f"Detected {total_defects} defect(s)."
                )

                df = pd.DataFrame(defect_data)

                st.subheader("Detected Defects")

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                # =============================================
                # DEFECT DISTRIBUTION
                # =============================================

                st.subheader("Defect Distribution")

                defect_counts = df["Defect"].value_counts()

                st.bar_chart(defect_counts)

            else:

                st.success(
                    "No defects detected in this image."
                )

        except Exception as e:

            st.error(
                f"An error occurred during detection: {e}"
            )

else:

    st.info(
        "Upload a PCB image to begin defect detection."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "PCB Defect Detection System | YOLOv8s | "
    "Six PCB Defect Categories"
)
