import cv2
import numpy as np
import mss

# === CONFIGURE HSV RANGE (color of enemy) ===
h_min, h_max = 0, 20
s_min, s_max = 100, 255
v_min, v_max = 100, 255

# === LOAD TEMPLATE IMAGE (enemy silhouette - white on black) ===
template = cv2.imread("enemy_template.png")
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
_, template_thresh = cv2.threshold(template_gray, 50, 255, cv2.THRESH_BINARY)
template_contours, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
enemy_contour = max(template_contours, key=cv2.contourArea)
template_match_gray = template_gray

# === DEFINE SCREEN REGION (center 300x300) ===
with mss.mss() as sct:
    monitor = sct.monitors[1]
    center_x = monitor['width'] // 2
    center_y = monitor['height'] // 2
    region = {
        'top': center_y - 150,
        'left': center_x - 150,
        'width': 300,
        'height': 300
    }

    while True:
        frame = np.array(sct.grab(region))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # HSV filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))

        # Contour detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = frame[y:y+h, x:x+w]

                try:
                    # Template matching
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    resized_roi = cv2.resize(roi_gray, template_match_gray.shape[::-1])
                    result = cv2.matchTemplate(resized_roi, template_match_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)

                    # Contour shape matching
                    match_score = cv2.matchShapes(cnt, enemy_contour, 1, 0.0)

                    # Combine both conditions
                    if max_val > 0.5 and match_score < 0.2:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Match: {max_val:.2f}", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except:
                    continue

        # Display output
        cv2.imshow("Live Detection", frame)
        cv2.imshow("HSV Mask", mask)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cv2.destroyAllWindows()
