import torch
import cv2
import numpy as np
from frame import preprocess_image

# Load YOLOv5 model (choose the model: yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load the video or use 0 for webcam
cap = cv2.VideoCapture('D:/Object_Detection_Project/yolov5/video.mp4')  # Change to your video path
if not cap.isOpened():
    raise IOError('Cannot open video source')

# Dictionary to count objects
object_counts = {}

# Processing each frame of the video
ctn = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Parse results
    detections = results.xyxy[0].cpu().numpy()  # Get the detections
    frame_height, frame_width, _ = frame.shape

    # Reset object counts
    object_counts.clear()

    
    # Process detections
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)  # Get box coordinates
        label = model.names[int(cls)]  # Get class label
        confidence = float(conf)

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Count the objects
        if label in object_counts:
            object_counts[label] += 1
        else:
            object_counts[label] = 1

    # Display the frame
    # output_test = preprocess_image(frame)
    # words = output_test.split(' ') 
    # n_words = 18
    # for val in range(0, len(words)-n_words, n_words):
    #     cv2.putText(frame, f"{' '.join(words[val:val+n_words])}", (40+val+5,40+val+5), 
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Object Detection", frame)
    if ctn > 1190:
        cv2.imwrite(f"write_op/{ctn}.jpg", frame)
    ctn+=1

    # for *box, conf, cls in detections:
    #     x1, y1, x2, y2 = map(int, box)  # Get box coordinates
    #     label = model.names[int(cls)]  # Get class label
    #     confidence = float(conf)

    #     # Draw bounding box and label on the frame
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(frame, f"{output_test}", (x1, y1 - 10), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #     # Count the objects
    #     if label in object_counts:
    #         object_counts[label] += 1
    #     else:
    #         object_counts[label] = 1

    # cv2.imshow("Object Detection1", frame)
    # Show object count on the console
    print("Object counts: ", object_counts)

    # Exit if 'Esc' is pressed
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
