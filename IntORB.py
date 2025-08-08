import cv2
import numpy as np
import mss
import os

# Path to reference images
REFERENCE_FOLDER = "./references"

# ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Load all reference images and compute descriptors
reference_data = []
for file in os.listdir(REFERENCE_FOLDER):
    path = os.path.join(REFERENCE_FOLDER, file)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    kp, des = orb.detectAndCompute(img, None)
    reference_data.append({"name": file, "image": img, "keypoints": kp, "descriptors": des})

if not reference_data:
    print("No valid reference images found!")
    exit()

# Matcher for ORB (Hamming norm, crossCheck for better accuracy)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Screen capture setup
with mss.mss() as sct:
    screen_width = sct.monitors[1]['width']
    screen_height = sct.monitors[1]['height']

    center_x = screen_width // 2
    center_y = screen_height // 2

    monitor_region = {
        'top': center_y - 150,
        'left': center_x - 150,
        'width': 300,
        'height': 300
    }

    while True:
        # Capture screen region
        frame = np.array(sct.grab(monitor_region))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        # ORB detection on current frame
        kp_frame, des_frame = orb.detectAndCompute(frame_gray, None)
        if des_frame is None:
            cv2.imshow("Live ORB Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        for ref in reference_data:
            matches = bf.match(ref["descriptors"], des_frame)
            matches = sorted(matches, key=lambda x: x.distance)

            # Only keep good matches (tweak threshold)
            good_matches = [m for m in matches if m.distance < 50]

            if len(good_matches) > 10:  # Minimum matches to consider a detection
                # Draw matches for debugging (optional)
                match_frame = cv2.drawMatches(ref["image"], ref["keypoints"], frame_gray, kp_frame, good_matches[:20], None, flags=2)

                # You can add homography to get bounding box
                src_pts = np.float32([ref["keypoints"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    h, w = ref["image"].shape
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 2)

                # Show optional match visualization window
                cv2.imshow(f"Matches - {ref['name']}", match_frame)

        cv2.imshow("Live ORB Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    
