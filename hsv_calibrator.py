import cv2
import numpy as np
import os

clicked_hsv_values = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img_bgr = param['image']
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        pixel = img_hsv[y, x]
        clicked_hsv_values.append(pixel)
        print(f"Clicked HSV: {pixel}")

def process_image(image_path):
    image = cv2.imread(image_path)
    clone = image.copy()
    cv2.imshow("Click on HEAD - Press ESC when done", clone)
    cv2.setMouseCallback("Click on HEAD - Press ESC when done", click_event, param={'image': clone})
    
    while True:
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break
    cv2.destroyAllWindows()

def calculate_hsv_bounds(hsv_values):
    hsv_array = np.array(hsv_values)
    h_min, s_min, v_min = np.min(hsv_array, axis=0)
    h_max, s_max, v_max = np.max(hsv_array, axis=0)
    return {
        'h_min': int(h_min),
        'h_max': int(h_max),
        's_min': int(s_min),
        's_max': int(s_max),
        'v_min': int(v_min),
        'v_max': int(v_max)
    }

if __name__ == "__main__":
    folder = "screenshots"  # Your folder with CS2 screenshots
    image_files = [os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("❌ No images found in folder.")
        exit()

    for image_path in image_files:
        print(f"\n🖼️ Processing: {os.path.basename(image_path)}")
        process_image(image_path)

    if clicked_hsv_values:
        hsv_range = calculate_hsv_bounds(clicked_hsv_values)
        print("\n✅ Final HSV Range:")
        for k, v in hsv_range.items():
            print(f"{k} = {v}")
    else:
        print("⚠️ No clicks registered.")
