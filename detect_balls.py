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
# 2) Approximate HSV color ranges
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

def detect_pool_balls(image_path):
    """
    Detects circles (balls) via HoughCircles, extracts each ball's color,
    and classifies them by standard 8-ball numbering.
    Returns:
        annotated_image : BGR image with labeled circles drawn
        balls_info      : list of (x, y, r, label, number)
        ball_mask       : mask of the balls
    """
    # 1. Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not open image at {image_path}")

    # We'll draw on a copy so we don't modify the original
    annotated = image.copy()

    # mask of the balls
    ball_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 2. Convert to grayscale & blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # find only the circles in "mask.png" over the original image
    table_mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
    gray = cv2.bitwise_and(gray, gray, mask=table_mask)

    # 3. Hough Circle detection
    #    Adjust these params to fit your image size and ball sizes
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=100,
        param2=25,
        minRadius=25,
        maxRadius=30
    )

    balls_info = []

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
            ball_label  = f"{color_name} #{ball_number}"

            # 7. Save info
            balls_info.append((x, y, r, ball_label, ball_number))

            # 8. Draw the circle & label on the annotated image
            cv2.circle(annotated, (x, y), r, (0, 255, 0), 2)
            cv2.putText(
                annotated, ball_label,
                (x - r, y - r - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )
    else:
        print("No circles detected by Hough transform.")

    # create a mask of the balls
    for (x, y, r, label, number) in balls_info:
        cv2.circle(ball_mask, (x, y), r, 255, -1)

    return annotated, balls_info, ball_mask

# -------------------------------------------------------------------------
#  Main script usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Change this to your image path
    image_path = "first_pics/1.jpeg"

    # extract colors to balls using select_roi
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not open image at {image_path}")
    
    # hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    """
    for color in COLOR_RANGES:
        print(f"Select ROI for {color} ball")
        roi = cv2.selectROI(hsv_image, False)
        cv2.destroyAllWindows()
        # color of the ball
        hsv_roi = hsv_image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        print(f"Selected HSV ROI")
        print(f"Average H: {np.mean(hsv_roi[:,:,0]):.2f}")
        print(f"Average S: {np.mean(hsv_roi[:,:,1]):.2f}")
        print(f"Average V: {np.mean(hsv_roi[:,:,2]):.2f}")
       """
    
    # choose 2 colors(by select ROI) and print their color ranges in hsv
    """
    print("Select ROI for two colors")
    roi1 = cv2.selectROI(hsv_image, False)
    cv2.destroyAllWindows()
    roi2 = cv2.selectROI(hsv_image, False)
    cv2.destroyAllWindows()
    hsv_roi1 = hsv_image[int(roi1[1]):int(roi1[1]+roi1[3]), int(roi1[0]):int(roi1[0]+roi1[2])]
    hsv_roi2 = hsv_image[int(roi2[1]):int(roi2[1]+roi2[3]), int(roi2[0]):int(roi2[0]+roi2[2])]
    print(f"Selected HSV ROI 1")
    print(f"Average H: {np.mean(hsv_roi1[:,:,0]):.2f}")
    print(f"Average S: {np.mean(hsv_roi1[:,:,1]):.2f}")
    print(f"Average V: {np.mean(hsv_roi1[:,:,2]):.2f}")
    print(f"Selected HSV ROI 2")
    print(f"Average H: {np.mean(hsv_roi2[:,:,0]):.2f}")
    print(f"Average S: {np.mean(hsv_roi2[:,:,1]):.2f}")
    print(f"Average V: {np.mean(hsv_roi2[:,:,2]):.2f}")
"""



    annotated_image, balls, balls_mask = detect_pool_balls(image_path)

    # Print results
    for (x, y, r, label, number) in balls:
        print(f"Detected {label} at ({x}, {y}), radius={r}")

    # Show final annotated image
    cv2.imshow("Detected Pool Balls", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # show the mask of the balls
    cv2.imshow("Mask of the balls", balls_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()