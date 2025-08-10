import cv2
import numpy as np
import mss
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/my_yolov8n/weights/best.pt")

# Screen capture object
sct = mss.mss()

# Monitor to capture (full screen)
monitor = sct.monitors[1]  # 1 = primary monitor

while True:
    # Grab screen
    screenshot = np.array(sct.grab(monitor))

    # Convert BGRA → BGR
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    # Run detection
    results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)

    # Draw detections on frame
    annotated_frame = results[0].plot()

    # Show on screen
    cv2.imshow("YOLOv8 Screen Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
