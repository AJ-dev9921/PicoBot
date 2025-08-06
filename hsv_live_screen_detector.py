import cv2
import numpy as np
import mss

# === Configure your HSV values here ===
h_min = 0
h_max = 20
s_min = 100
s_max = 255
v_min = 100
v_max = 255

# Screen capture region (full screen or a cropped window)
monitor_region = sct.monitors[1]

# Start screen capture
with mss.mss() as sct:
    while True:
        frame = np.array(sct.grab(monitor_region))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)

        # Draw rectangles around detected areas
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show screen detection
        cv2.imshow("Live HSV Detection", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()
