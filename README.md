# Object Detection in Images using YOLOv10


<!-- Tags for Google Colab -->
<a href="">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>

<!-- Tags for Streamlit -->
<a href="">
        <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open In Streamlit"/>
    </a>

## Introduction
Object Detection Model Using YOLOv10 for detecting objects in images.

* Model Variants:
  - YOLOv10-N: Nano version for extremely resource-constrained environments.
  - YOLOv10-S: Small version balancing speed and accuracy.
  - YOLOv10-M: Medium version for general-purpose use.
  - YOLOv10-B: Balanced version with increased width for higher accuracy.
  - YOLOv10-L: Large version for higher accuracy at the cost of increased computational resources.
  - YOLOv10-X: Extra-large version for maximum accuracy and performance.

## Pretrained Models
YOLOv10 outperforms previous YOLO versions and other state-of-the-art models in terms of accuracy and efficiency. For example, YOLOv10-S is 1.8x faster than RT-DETR-R18 with similar AP on the COCO dataset, and YOLOv10-B has 46% less latency and 25% fewer parameters than YOLOv9-C with the same performance.

| Model     | Input Size | APval | FLOPs (G) | Latency (ms) |
|-----------|------------|-------|-----------|--------------|
| YOLOv10-N | 640        | 38.5  | 6.7       | 1.84         |
| YOLOv10-S | 640        | 46.3  | 21.6      | 2.49         |
| YOLOv10-M | 640        | 51.1  | 59.1      | 4.74         |
| YOLOv10-B | 640        | 52.5  | 92.0      | 5.74         |
| YOLOv10-L | 640        | 53.2  | 120.3     | 7.28         |
| YOLOv10-X | 640        | 54.4  | 160.4     | 10.70        |

* Latency measured with TensorRT FP16 on T4 GPU.

## Usage
Use the following command to run the model:
1. Run using Google Colab button.
2. Run using Streamlit button.
3. Clone the repository and run the following command:
4. `pip install -r requirements.txt`
5. After installing the required libraries, run the following command:
```bash
streamlit run app.py
```

## References

[YOLOv10](https://docs.ultralytics.com/models/yolov10/)
