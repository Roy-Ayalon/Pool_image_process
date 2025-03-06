import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import mode


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
# 2) Approximate HSV color ranges
# -------------------------------------------------------------------------
COLOR_RANGES = {
    "yellow":   ((16,  147,  224),  (36, 167, 244)),  # new
    "brown":    ((0,   157,  150),  (40, 170, 175)),  # new
    "blue":     ((80,  130,  160),  (115, 220, 210)), # new
    "red":      ((115,   125,  210),  (179, 230, 255)),  # new
#    "orange":   ((0,  160,  230),  (40, 210, 270)),  # NOT GOOD
    "green":    ((60,  130,  130),  (110, 230, 200)),  # new
#    "purple":   ((90, 70,  100),  (150, 170, 180)), # new
    "black":    ((30,   120,    0),    (85, 190, 50))    # new
}

WHITE_RANGE = {
    "white":    ((30,   0,    200),  (80, 70, 255)),  # hue ~0-180, sat ~0-50, val ~200-255
}

def classify_ball_color(hsv_ball_roi, COLOR_RANGES):
    """
    Given an HSV image of a single ball (already cropped circularly),
    decide if it's black, white, or one of the colors in COLOR_RANGES.
    Uses:
      1) Average H, S, V (like your old approach),
      2) Coverage approach (pixel count) to pick the best color.

    Returns:
        A string (e.g., "black", "white", "blue", "red", etc.)
    """

    # # 1) Compute average H, S, V (like the old function)
    # avg_h = np.mean(hsv_ball_roi[:, :, 0])
    # avg_s = np.mean(hsv_ball_roi[:, :, 1])
    # avg_v = np.mean(hsv_ball_roi[:, :, 2])

    # 2) Among the color ranges, find which covers the most pixels
    best_color = None
    best_count = 0

    h, w = hsv_ball_roi.shape[:2]  # Not essential, but kept for reference

    for color_name, (lower, upper) in COLOR_RANGES.items():
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)

        # Make a mask for this color’s HSV range
        mask = cv2.inRange(hsv_ball_roi, lower_np, upper_np)
        count = cv2.countNonZero(mask)

        # If this color covers more pixels than the current best, 
        # we tentatively pick it as best_color...
        if count > best_count:
            best_color = color_name
            best_count = count

    # 3) Fallback: If no color or best_color is None, default to "white"
    # # (Your old code used “white” as fallback)
    # if best_color is None:
    #     return "white"

    # 4) (Optional) We can also ensure avg H,S,V fits into that color range:
    #    But since your old approach was correct “most of the time,”
    #    we can just return best_color from coverage.
    #
    #    If you want to confirm that (avg_h, avg_s, avg_v) is in that color range:
    #        lower, upper = COLOR_RANGES[best_color]
    #        if not (lower[0] <= avg_h <= upper[0] and
    #                lower[1] <= avg_s <= upper[1] and
    #                lower[2] <= avg_v <= upper[2]):
    #            best_color = "white"  # or "unknown"

    return best_color, best_count

def extract_circular_roi(image, x, y, r):
    """
    Extracts a circular ROI from an image given the center (x, y) and radius (r).
    
    Parameters:
        image (ndarray): Input image.
        x (int): Center x-coordinate of the ball.
        y (int): Center y-coordinate of the ball.
        r (int): Radius of the ball.

    Returns:
        circular_roi (ndarray): Circular ROI with the background set to black.
    """

    # Define the bounding square for the ball
    x1, x2 = max(0, x - r), min(image.shape[1], x + r)
    y1, y2 = max(0, y - r), min(image.shape[0], y + r)

    # Crop the square ROI
    square_roi = image[y1:y2, x1:x2].copy()

    # Create a circular mask **that fits the cropped region size**
    mask = np.zeros(square_roi.shape[:2], dtype=np.uint8)
    
    # Compute the correct center of the mask
    center_x = (x2 - x1) // 2
    center_y = (y2 - y1) // 2

    # Draw a filled circle mask
    cv2.circle(mask, (center_x, center_y), min(r, center_x, center_y), 255, -1)

    # Apply the mask to extract the circular region
    circular_roi = cv2.bitwise_and(square_roi, square_roi, mask=mask)

    return circular_roi
import cv2
import numpy as np

def detect_pool_balls(image, board_contour):
    """
    Detects both white and colored balls on the table, returning only the single largest white ball.
    
    Returns:
        - annotated: BGR image with recognized circles (colored and ONE white) drawn
        - balls_info: list of (x, y, r, label, number) for all recognized balls (including 1 white ball)
        - ball_mask: binary mask of recognized balls
        - contour_balls: list of ball contours
        - binary: binary image for debug
        - white_ball: tuple (x, y, r) for the single largest white ball or None
    """

    annotated = image.copy()
    ball_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Restrict detection inside the board
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [board_contour], -1, 255, -1)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Hough Circle detection
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=10,
        param1=50,
        param2=30,
        minRadius=14,
        maxRadius=22
    )

    balls_info = []         # For all recognized (non-white) colored balls
    contour_balls = []      # For storing ball contour circles
    binary = np.zeros_like(image)

    white_ball = detect_white_ball(image, board_contour)

    if circles is not None and len(circles) > 0:
        circles = np.uint16(np.around(circles)).astype(np.int32)

        for (x, y, r) in circles[0]:
            # Check boundaries
            if (x - r < 0) or (y - r < 0) or (x + r >= image.shape[1]) or (y + r >= image.shape[0]):
                continue

            # Extract ball ROI
            ball_roi = extract_circular_roi(image, x, y, r)
            hsv_roi = cv2.cvtColor(ball_roi, cv2.COLOR_BGR2HSV)

            # Classify ball color
            color_name, best_count = classify_ball_color(hsv_roi, COLOR_RANGES)
            # if color_name is None:
            #     if white_ball is not None:
            #         wx, wy, wr = white_ball
            #         if (x,y,r) == (wx, wy, wr):
            #             continue
            #     balls_info.append((x, y, r, '', ''))
            #     continue  # Skip if not recognized at all

            if color_name is None:
                continue  # Skip if not recognized at all

            # For colored ball
            ball_number = COLOR_TO_BALL.get(color_name, -1)
            if ball_number == -1:
                continue  # Unknown color => skip

            # Save recognized colored ball
            balls_info.append((x, y, r, color_name, ball_number))

            # Draw on annotated and binary
            cv2.circle(binary, (x, y), r, (255, 0, 0), 1)
            cv2.circle(annotated, (x, y), r, (0, 255, 0), 2)

            # Draw contour on ball_mask
            cv2.circle(ball_mask, (x, y), r, 255, -1)
            contour_balls.append(True)  # We just record something; optionally store (x,y,r)

    return annotated, balls_info, ball_mask, contour_balls, binary




def detect_white_ball(image, board_contour):
    
    annotated = image.copy()
    ball_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Restrict detection inside the board
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [board_contour], -1, 255, -1)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Hough Circle detection
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=10,
        param1=50,
        param2=30,
        minRadius=14,
        maxRadius=22
    )

    contour_balls = []      # For storing ball contour circles
    binary = np.zeros_like(image)
    white_candidates = []   # Collect all white ball candidates here

    if circles is not None and len(circles) > 0:
        circles = np.uint16(np.around(circles)).astype(np.int32)

        for (x, y, r) in circles[0]:
            # Check boundaries
            if (x - r < 0) or (y - r < 0) or (x + r >= image.shape[1]) or (y + r >= image.shape[0]):
                continue

            # Extract ball ROI
            ball_roi = extract_circular_roi(image, x, y, r)
            hsv_roi = cv2.cvtColor(ball_roi, cv2.COLOR_BGR2HSV)

            # Classify ball color
            color_name, best_count = classify_ball_color(hsv_roi, WHITE_RANGE)
            

            # If it's white, store candidate and skip for now
            if color_name == "white":
                white_candidates.append((x, y, r, best_count))
                continue

    # ----- Now pick exactly one largest white ball among the candidates -----
    white_ball = None
    # largest_radius = 0
    most_white = 0

    for (wx, wy, wr, best_count) in white_candidates:
        if most_white < best_count:
            most_white = best_count
            white_ball = (wx, wy, wr)

    # If we found a white ball, add it to final output
    if white_ball:
        wx, wy, wr = white_ball

    return  white_ball

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