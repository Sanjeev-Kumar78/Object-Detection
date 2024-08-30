import streamlit as st
from Detector import Detector
from PIL import Image
import numpy as np

# Set up the Streamlit app
st.title("YOLOv10 Object Detection App")
st.write("Upload an image to perform object detection using YOLOv10")

# Sidebar for selecting model and threshold
modelName = st.sidebar.selectbox(
    "Select YOLOv10 Model",
    ["yolov10n.pt", "yolov10s.pt", "yolov10m.pt","yolov10b.pt", "yolov10l.pt", "yolov10x.pt"],
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
    st.write("Running YOLOv10...")
    result_image = detector.createBoundingBox(image_np, threshold)

    # Display the result image
    st.image(result_image, caption="Detected Objects", use_column_width=True)

else:
    st.write("Please upload an image to get started.")
