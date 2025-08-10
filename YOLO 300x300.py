import cv2
import numpy as np
import mss
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")

# Screen capture setup
sct = mss.mss()

# Monitor info (1 = primary monitor)
monitor_info = sct.monitors[1]
screen_width = monitor_info["width"]
screen_height = monitor_info["height"]

# Calculate center capture box (300x300)
box_width, box_height = 300, 300
left = (screen_width - box_width) // 2
top = (screen_height - box_height) // 2
capture_region = {
    "top": top,
    "left": left,
    "width": box_width,
    "height": box_height
}

while True:
    # Capture only the center region
    screenshot = np.array(sct.grab(capture_region))

    # Convert BGRA → BGR
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    # Run YOLO detection
    results = model.predict(frame, imgsz=640, conf=0.25, verbose=False, show=False)

    # Annotated frame
    annotated_frame = results[0].plot()

    # Show results (small window)
    cv2.imshow("Center Detection", annotated_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
