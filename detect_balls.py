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
    "blue":     ((80,  200,  108),  (114, 246, 170)), # new
    "red":      ((158,   195,  167),  (179, 255, 187)),  # new
    "orange":   ((0,  160,  230),  (40, 210, 270)),  # NOT GOOD
    "green":    ((75,  200,  85),  (95, 230, 115)),  # new
    "purple":   ((128, 125,  100),  (158, 180, 170)), # new
    "white":    ((60,   0,    230),  (100, 20, 255)),  # hue ~0-180, sat ~0-50, val ~200-255
    "black":    ((30,   120,    0),    (85, 190, 50))    # new
}

def classify_ball_color(hsv_ball_roi):
    """
    Given an HSV image of a single ball (already cropped circularly),
    decide if it's black, white, or one of the colors in COLOR_RANGES.
    Uses:
      1) Average H, S, V (like your old approach),
      2) Coverage approach (pixel count) to pick the best color.

    Returns:
        A string (e.g., "black", "white", "blue", "red", etc.)
    """

    # 1) Compute average H, S, V (like the old function)
    avg_h = np.mean(hsv_ball_roi[:, :, 0])
    avg_s = np.mean(hsv_ball_roi[:, :, 1])
    avg_v = np.mean(hsv_ball_roi[:, :, 2])

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
    # (Your old code used “white” as fallback)
    if best_color is None:
        return "white"

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

    return best_color

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
        minRadius=15,
        maxRadius=20
    )

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
            ball_roi = extract_circular_roi(image, x, y, r)
            hsv_roi = cv2.cvtColor(ball_roi, cv2.COLOR_BGR2HSV)

            # 6. Classify color
            color_name = classify_ball_color(hsv_roi)

        
            # Use the color name to determine if it's solid or striped
            ball_number = COLOR_TO_BALL[color_name]
            ball_label  = f"{color_name}"

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



def detect_white_ball(frame, board_contour, min_radius=15, max_radius=25):
    """Detects the white cue ball on the pool table by selecting the largest detected white ball."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define HSV range for white color (tuned for typical lighting conditions)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([140, 90, 255], dtype=np.uint8)
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

