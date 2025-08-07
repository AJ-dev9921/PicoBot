import cv2
import numpy as np
import mss
import os

# Load the game character images from the folder
image_folder = 'game_characters'  # Folder containing character images
images = []
image_names = []

# Check if the folder exists and has images
if not os.path.exists(image_folder):
    print(f"Error: Folder '{image_folder}' not found!")
    exit()

if len(os.listdir(image_folder)) == 0:
    print(f"Error: No images found in '{image_folder}'!")
    exit()

for filename in os.listdir(image_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(image_folder, filename), cv2.IMREAD_GRAYSCALE)
        images.append(img)
        image_names.append(filename)

# Initialize ORB detector
orb = cv2.ORB_create()

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

    # Create the window and the trackbar for threshold
    cv2.namedWindow('ORB Detection')
    cv2.createTrackbar('Accuracy Threshold', 'ORB Detection', 30, 100, lambda x: None)

    while True:
        # Capture the defined 300x300 region
        frame = np.array(sct.grab(monitor_region))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find keypoints and descriptors in the frame
        kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

        # Check if descriptors were detected
        if des_frame is None:
            continue  # Skip this frame if no descriptors were found

        # Get the current threshold value from the trackbar
        threshold = cv2.getTrackbarPos('Accuracy Threshold', 'ORB Detection')

        # Loop through each loaded character image
        for img, name in zip(images, image_names):
            # Find keypoints and descriptors in the character image
            kp_img, des_img = orb.detectAndCompute(img, None)

            # Check if descriptors were detected
            if des_img is None:
                continue  # Skip this character image if no descriptors were found

            # Use the BFMatcher to match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des_frame, des_img)

            # Sort matches based on distance (lower distance = better match)
            matches = sorted(matches, key=lambda x: x.distance)

            # Handle empty matches to avoid division by zero
            if len(matches) > 0:
                match_distance = sum([m.distance for m in matches[:10]]) / len(matches[:10])  # Average match distance
            else:
                match_distance = float('inf')  # No matches, set distance to infinity

            # If the match distance is below the threshold, consider it a match
            if match_distance < threshold:
                cv2.putText(frame, f'Match: {name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f'No match for {name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the result
        cv2.imshow("ORB Detection", frame)

        # Exit loop if ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release resources
    cv2.destroyAllWindows()
