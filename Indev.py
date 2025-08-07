import cv2
import numpy as np
import mss
import os

# === Load reference images ===
ref_folder = 'refs'
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # Use KNN

ref_data = []
for filename in os.listdir(ref_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(ref_folder, filename)
        ref_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(ref_img, None)
        if des is not None:
            ref_data.append({'name': filename, 'image': ref_img, 'kp': kp, 'des': des})
        else:
            print(f"⚠️ No descriptors found in {filename}")

if not ref_data:
    print("❌ No valid reference images found.")
    exit()

# === Screen capture ===
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
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        kp2, des2 = orb.detectAndCompute(frame_gray, None)
        best_match = None
        best_score = 0

        if des2 is not None:
            for ref in ref_data:
                matches = bf.knnMatch(ref['des'], des2, k=2)

                # Apply Lowe’s ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                if len(good_matches) > best_score:
                    best_score = len(good_matches)
                    best_match = {
                        'name': ref['name'],
                        'matches': good_matches,
                        'ref_img': ref['image'],
                        'ref_kp': ref['kp'],
                        'frame_kp': kp2
                    }

        # Show results
        if best_match and best_score >= 10:
            print(f"🔍 Best Match: {best_match['name']} | Matches: {best_score}   ", end="\r")
            matched = cv2.drawMatches(
                best_match['ref_img'], best_match['ref_kp'],
                frame_bgr, best_match['frame_kp'],
                best_match['matches'], None, flags=2
            )
            cv2.imshow("ORB Match", matched)
        else:
            cv2.imshow("ORB Match", frame_bgr)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    
