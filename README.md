# 🧠 Object Counting and Face Detection with YOLOv8 & OpenCV

A computer vision project that combines **YOLOv8** object detection with **OpenCV Haar Cascade** face detection to identify, count, and analyze people within images. The project is designed and tested in **Google Colab**, making it easy to run without complex local setup.

## 🚀 Overview

This application detects objects in an image using the **YOLOv8** model and then performs additional **face detection** within each detected person. The result is an annotated image showing detected objects, people, and faces, along with object counts.

## ✨ Features

* Upload and process images directly in **Google Colab**.
* Detect multiple object classes using **YOLOv8**.
* Count the total number of detected objects.
* Identify and count all detected **persons**.
* Perform **face detection** within person regions using OpenCV's Haar Cascade classifier.
* Draw bounding boxes and labels around detected objects and faces.
* Save the processed image with annotations.
* Display the final output image within the notebook.

## 📸 Sample Output

![Sample Output](output_with_boxes.jpg)

## 🛠️ Technologies and Libraries

* **Python**
* **OpenCV**
* **Ultralytics YOLOv8 (`yolov8n.pt`)**
* **Google Colab**
* **Haar Cascade Face Detection**

## 📦 Installation

Install the required dependencies:

```bash
pip install ultralytics opencv-python
```

> Note: OpenCV (`cv2`) is already available in Google Colab, so no additional setup is typically required.

## 📂 Project Structure

```text
.
├── counting_objects_face_detection.ipynb
├── output_with_boxes.jpg
└── README.md
```

## ⚙️ Workflow

1. Upload an image to the Colab environment.
2. Load the YOLOv8 model.
3. Detect and classify objects in the image.
4. Count all detected objects.
5. Filter detections belonging to the **person** class.
6. Apply Haar Cascade face detection to each detected person.
7. Draw bounding boxes and labels on the image.
8. Save the annotated image as `output_with_boxes.jpg`.
9. Display the final processed image.

## 🔍 Code Example

```python
from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Run object detection
results = model(image)
detections = results[0]

# Count detected objects
total_objects = len(detections.boxes)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
```

## 📈 Future Enhancements

* Replace Haar Cascades with modern deep learning-based face detectors.
* Support real-time webcam and video stream processing.
* Build an interactive web application using **Gradio** or **Streamlit**.
* Generate detailed detection statistics and analytics.
* Extend object counting capabilities to specific categories such as vehicles, animals, and safety equipment.

## 👨‍💻 Author

**Naqeeb Ullah**

Connect on LinkedIn:
https://www.linkedin.com/in/naqeeb-ullah-bb46a9150/

---

If you find this project useful, consider giving it a ⭐ and contributing to future improvements.
