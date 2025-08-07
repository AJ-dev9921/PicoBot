import cv2
import numpy as np
import mss
import os

# === Load all reference images from 'refs' folder ===
ref_folder = 'refs'
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

ref_data = []

for filename in os.listdir(ref_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(ref_folder, filename)
        ref_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(ref_img, None)
        if des is not None:
            ref_data.append({'name': filename, 'image': ref_img, 'kp': kp, 'des': des})
        else:
            print(f"⚠️ Warning: No descriptors found in {filename}")

if not ref_data:
    print("❌ No valid reference images found.")
    exit()

# === Screen region config ===
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
        best_score = float('inf')

        if des2 is not None:
            for ref in ref_data:
                matches = bf.match(ref['des'], des2)
                matches = sorted(matches, key=lambda x: x.distance)

                # Score based on sum of top distances
                top_matches = matches[:20]
                score = sum([m.distance for m in top_matches])

                if score < best_score:
                    best_score = score
                    best_match = {
                        'name': ref['name'],
                        'matches': top_matches,
                        'ref_img': ref['image'],
                        'ref_kp': ref['kp'],
                        'frame_kp': kp2
                    }

        # Show match
        if best_match:
            print(f"🔍 Best Match: {best_match['name']} (Score: {int(best_score)})", end='\r')

            matched = cv2.drawMatches(
                best_match['ref_img'], best_match['ref_kp'],
                frame_bgr, best_match['frame_kp'],
                best_match['matches'], None, flags=2
            )
            cv2.imshow("Best Match", matched)
        else:
            cv2.imshow("Best Match", frame_bgr)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    
