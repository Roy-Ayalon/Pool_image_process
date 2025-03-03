import cv2
import numpy as np
from matplotlib import pyplot as plt

import cv2
import numpy as np

# -------------------------------------------------------------------------
# 1) Color -> Ball number map (based on standard 8-ball pool)
# -------------------------------------------------------------------------
COLOR_TO_BALL = {
    "yellow": 1,
    "blue": 2,
    "red": 3,
    "purple": 4,
    "orange": 5,
    "green": 6,
    "brown": 7,  # Some sets call it maroon or brown
    "black": 8,           # 8-ball
    "yellow": 9, 
    "blue": 10,
    "red": 11,
    "purple": 12,
    "orange": 13,
    "green": 14,
    "brown": 15,
    "white": 0        # 
}

# -------------------------------------------------------------------------
# 2) Approximate HSV color ranges MIKI the KIng
#    These are VERY approximate and must be tuned for your lighting.
#    For example, "red" might appear around hue=0 or hue~170-180.
# -------------------------------------------------------------------------
COLOR_RANGES = {
    "yellow":   ((20,  220,  235),  (40, 240, 255)),  # hue ~30, sat ~230, val ~230
    "brown":    ((0,   230,  120),  (20, 250, 140)),  # hue ~10, sat ~240, val ~130
    "blue":     ((90,  170,  70),  (120, 210, 110)), # hue ~100-110, sat ~190, val ~90
    "red":      ((0,   220,  120),  (10, 255, 160)),  # hue ~0-10, sat ~230, val ~140
    "orange":   ((10,  230,  230),  (20, 255, 255)),  # hue ~15, sat ~240, val ~240
    "green":    ((50,  190,  70),  (85, 210, 100)),  # hue ~60-70, sat ~200, val ~85
    "purple":   ((150, 160,  120),  (175, 180, 150)), # hue ~140, sat ~150, val ~150
    "white":    ((60,   0,    230),  (100, 20, 255)),  # hue ~0-180, sat ~0-50, val ~200-255
    "black":    ((20,   115,    10),    (50, 135, 40))    # hue ~0-180, sat ~0-255, val ~0-30
}

def classify_ball_color(hsv_ball_roi):
    """
    Given an HSV image of a single ball,
    decide if it's black, white, or one of the colors in COLOR_RANGES.
    Returns a string like "black", "white", or "blue", "red", etc.
    """
    # Compute average H, S, V
    avg_h = np.mean(hsv_ball_roi[:,:,0])
    avg_s = np.mean(hsv_ball_roi[:,:,1])
    avg_v = np.mean(hsv_ball_roi[:,:,2])

    # Otherwise, find which color range covers the most pixels
    best_color = None
    best_count = 0
    h, w = hsv_ball_roi.shape[:2]

    for color_name, (lower, upper) in COLOR_RANGES.items():
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)

        mask = cv2.inRange(hsv_ball_roi, lower_np, upper_np)
        count = cv2.countNonZero(mask)
        if count > best_count:
            best_count = count
            best_color = color_name

    if best_color is None:
        return "white"  # fallback
    return best_color

def detect_pool_balls(image, board_contour):
    """
    Detects circles (balls) via HoughCircles, extracts each ball's color,
    and classifies them by standard 8-ball numbering.
    Returns:
        annotated_image : BGR image with labeled circles drawn
        balls_info      : list of (x, y, r, label, number)
        ball_mask       : mask of the balls
    """
    # # 1. Load the image
    # image = cv2.imread(image_path)
    # if image is None:
    #     raise IOError(f"Could not open image at {image_path}")

    # We'll draw on a copy so we don't modify the original
    annotated = image.copy()

    # mask of the balls
    ball_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 2. Convert to grayscale & blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = cv2.medianBlur(gray, 5)

    # find only the circles in contour over the original image
    mask = np.zeros_like(gray)
    #gray_image = y_i.astype(np.uint8)
    cv2.drawContours(mask, [board_contour], -1, 255, -1)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    
    # 3. Hough Circle detection
    #    Adjust these params to fit your image size and ball sizes
    #! changes, work with MAC
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=10,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=20
    )

    # circles = cv2.HoughCircles(
    #     gray,
    #     cv2.HOUGH_GRADIENT,
    #     dp=2,
    #     minDist=15,
    #     param1=5,
    #     param2=35,
    #     minRadius=11,
    #     maxRadius=13
    # )

    balls_info = []
    contour_balls = []
    binary = np.zeros_like(image)
    if circles is not None and len(circles) > 0:
        # Convert to int32 (avoid uint16 overflow when subtracting)
        circles = np.uint16(np.around(circles))
        circles = circles.astype(np.int32)  # each circle is now (x, y, r)
        for (x, y, r) in circles[0]:
            # 4. Boundary check: skip circles that go out of the image
            if (x - r < 0) or (y - r < 0) or (x + r >= image.shape[1]) or (y + r >= image.shape[0]):
                continue

            # 5. Extract ball ROI
            ball_roi = image[y - r : y + r, x - r : x + r]
            hsv_roi = cv2.cvtColor(ball_roi, cv2.COLOR_BGR2HSV)

            # 6. Classify color
            color_name = classify_ball_color(hsv_roi)

        
            # Use the color name to determine if it's solid or striped
            ball_number = COLOR_TO_BALL[color_name]
            ball_label  = f""

            # 7. Save info
            balls_info.append((x, y, r, ball_label, ball_number))

            # 8. Draw the circle & label on the annotated image
            cv2.circle(binary, (x, y), r, (255, 0, 0), 1)
            #cv2.putText(
            #    annotated, ball_label,
            #    (x - r, y - r - 5),
            #    cv2.FONT_HERSHEY_SIMPLEX,
            #    0.6, (255, 0, 0), 2
            #)

            # add contour of the ball to the list
            contour_balls.append(cv2.circle(ball_mask, (x, y), r, 255, -1))
    else:
        print("No circles detected by Hough transform.")

    

    # create a mask of the balls
    for (x, y, r, label, number) in balls_info:
        cv2.circle(ball_mask, (x, y), r, 255, -1)

    return annotated, balls_info, ball_mask, contour_balls, binary



def detect_white_ball(frame, board_contour, min_radius=10, max_radius=25):
    """Detects the white cue ball on the pool table by selecting the largest detected white ball."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define HSV range for white color (tuned for typical lighting conditions)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 50, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Mask only inside the board
    mask = np.zeros_like(mask_white)
    cv2.drawContours(mask, [board_contour], -1, 255, -1)
    mask_white = cv2.bitwise_and(mask_white, mask_white, mask=mask)
    
    # Find contours of possible white balls
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_radius = 0
    largest_ball = None
    
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if min_radius <= radius <= max_radius and radius > largest_radius:  # Select only the largest valid white ball
            largest_radius = radius
            largest_ball = (int(x), int(y), int(radius))
    
    # Draw only the largest detected white ball if within valid range
    if largest_ball:
        x, y, radius = largest_ball
        cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)  # Draw white ball in red
        cv2.putText(frame, "White Ball", (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    

    return frame, largest_ball

def main():
    cap = cv2.VideoCapture(0)  # Adjust camera index if needed
    if not cap.isOpened():
        print("Error opening camera")
        return

    # Capture an initial frame for ROI selection.
    ret, frame = cap.read()
    if not ret:
        print("Error reading from camera")
        return

    # Select ROI covering the tip of the stick to determine its color.
    lower_color, upper_color = select_roi_and_get_color(frame)

    print("Starting stick detection. Press ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to HSV and generate a binary mask for the selected color.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, lower_color, upper_color)

        # Pass the original frame and the HSV mask into the stick detector.
        result_img, refined_line = detect_stick(frame, mask_hsv)

        # Display both the original frame and the result from stick detection.
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Stick Detection", result_img)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # Press ESC to exit.
            break

    cap.release()
    cv2.destroyAllWindows()



