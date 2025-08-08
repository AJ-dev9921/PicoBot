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

# Get the screen's width and height
with mss.mss() as sct:
    screen_width = sct.monitors[1]['width']
    screen_height = sct.monitors[1]['height']

    # Calculate the center of the screen
    center_x = screen_width // 2
    center_y = screen_height // 2

    # Define the 300x300 region around the center
    monitor_region = {
        'top': center_y - 150,  # 150 pixels above the center
        'left': center_x - 150, # 150 pixels to the left of the center
        'width': 300,           # 300 pixels wide
        'height': 300           # 300 pixels tall
    }

    while True:
        # Capture the defined 300x300 region
        frame = np.array(sct.grab(monitor_region))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Convert to HSV and create mask
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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show results
        cv2.imshow("Live HSV Detection", frame)
        cv2.imshow("Mask", mask)

        # Break loop on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
