# Object_Detection_Project

This project demonstrates object detection using YOLOv5.

## Setup Instructions

To set up the project, follow these steps:

1.  **Install Python Requirements:**
    Navigate to the project root directory and install the required Python packages:
    ```bash
    pip install -r Object-Detection-Project/requirements.txt
    pip install -r Object-Detection-Project/yolov5/requirements.txt
    ```

## Running Object Detection

To run object detection on the sample images:

1.  **Execute the Detection Script:**
    Run the `detect.py` script located in the `yolov5` subdirectory. This command will use the default `yolov5s.pt` weights and process the images in `yolov5/data/images`.
    ```bash
    python Object-Detection-Project/yolov5/detect.py --weights Object-Detection-Project/yolov5/yolov5s.pt --source Object-Detection-Project/yolov5/data/images
    ```

2.  **View Results:**
    The detection results (images with bounding boxes) will be saved in a new directory under `Object-Detection-Project/yolov5/runs/detect/`. For example, `Object-Detection-Project/yolov5/runs/detect/exp`.
