import cv2, os
import numpy as np
from ultralytics import YOLO  # Import YOLO from Ultralytics

np.random.seed(123)

class Detector:
    def __init__(self):
        self.classesList = []
        self.colorList = []
        self.model = None

    def downloadModel(self, modelName):
        try:
            # Define the model directory
            model_dir = "Model"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)  # Create the "Model" folder if it doesn't exist
            
            # Construct the path where the model should be saved
            model_path = os.path.join(model_dir, modelName)
            
            if not os.path.exists(model_path):  # Download only if the model isn't already downloaded
                print(f"Downloading YOLO model: {modelName}")
                self.model = YOLO(model_path)  # Load the YOLO model and save it to the specified path
            else:
                print(f"Model {modelName} already exists. Loading from {model_path}")
                self.model = YOLO(model_path)  # Load the model from the saved path
            
            # Set the class names and colors
            self.classesList = self.model.names
            self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
            print(f"Model {modelName} loaded successfully")
        
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")

    def createBoundingBox(self, image, threshold=0.5):
        try:
            bboxImage = image.copy()
            results = self.model(image)  # Perform inference using YOLOv10

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Extract coordinates and other information from YOLOv10's result
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    conf = box.conf[0]  # Confidence score
                    class_idx = int(box.cls[0])  # Class index

                    if conf >= threshold:
                        # Get the class label and color dynamically
                        class_label = self.classesList[class_idx] if class_idx < len(self.classesList) else f"ID {class_idx}"
                        color = self.colorList[class_idx]

                        display_text = f"{class_label}: {conf:.2f}"
                        cv2.rectangle(bboxImage, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(bboxImage, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            return bboxImage
        except Exception as e:
            print(f"An error occurred while creating bounding boxes: {e}")
            return None

    def predictImage(self, imagePath, threshold=0.5):
        try:
            img = cv2.imread(imagePath)
            if img is None:
                print(f"Error: Unable to read image from path {imagePath}")
                return
            elif img is not None:
                bboxImage = self.createBoundingBox(img, threshold)
            else:
                print(f"Error: Unable to read image from path {imagePath}")
            if bboxImage is not None:
                cv2.imshow("Detected Image", bboxImage)
                cv2.imwrite("output.jpg", bboxImage)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Error: Bounding box image is None.")
        except Exception as e:
            print(f"An error occurred while predicting the image: {e}")
            return None
