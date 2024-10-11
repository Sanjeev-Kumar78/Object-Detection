import streamlit as st
from Detector import Detector
from PIL import Image
import numpy as np

# Set up the Streamlit app
st.title("YOLO Object Detection App")
st.write("Upload an image to perform object detection using YOLO")

# Sidebar for selecting YOLO version
yolo_version = st.sidebar.selectbox(
    "Select YOLO Version",
    ["YOLOv10", "YOLOv11"],
    index=1
)

# Sidebar for selecting model and threshold
if yolo_version == "YOLOv10":
    modelName = st.sidebar.selectbox(
        "Select YOLOv10 Model",
        ["yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt"],
        index=0
    )
else:
    modelName = st.sidebar.selectbox(
        "Select YOLOv11 Model",
        ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11b.pt", "yolo11l.pt", "yolo11x.pt"],
        index=0
    )

threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.05)

# Image uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Convert the uploaded image to an OpenCV format
    image = Image.open(uploaded_image)
    image_np = np.array(image.convert('RGB'))  # Convert to RGB format

    # Initialize the Detector
    detector = Detector()
    detector.downloadModel(modelName)

    # Predict the image
    st.write(f"Running {yolo_version}...")
    result_image = detector.createBoundingBox(image_np, threshold)

    # Display the result image
    st.image(result_image, caption="Detected Objects", use_column_width=True)

else:
    st.write("Please upload an image to get started.")
