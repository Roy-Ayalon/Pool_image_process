import cv2
import numpy as np
from matplotlib import pyplot as plt

def detect_board(frame, debug=False):
    """
    Detect the board (table) using the LAB A channel.
    Returns the largest contour (assumed board) and a binary mask (filled board region).
    """
    lab_image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab_image)
    A_normalized = cv2.normalize(A, None, 0, 255, cv2.NORM_MINMAX)
    
    if debug:
        cv2.imshow("Normalized A Channel", A_normalized)
        cv2.waitKey(0)
    
    threshold = 70
    hist, bins = np.histogram(A_normalized.flatten(), bins=256, range=(0, 255))
    max_x = np.argmax(hist[:threshold])
    threshold_x = 25
    binary_mask = np.where((A_normalized >= (max_x - threshold_x)) &
                           (A_normalized <= (max_x + threshold_x)),
                           255, 0).astype(np.uint8)
    
    if debug:
        plt.plot(hist)
        plt.show()
        cv2.imshow("Binary Mask", binary_mask)
        cv2.waitKey(0)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if debug:
            print("No contours found.")
        return None, None
    largest_contour = max(contours, key=cv2.contourArea)
    
    board_mask = np.zeros_like(binary_mask)
    cv2.fillPoly(board_mask, [largest_contour], 255)
    
    return largest_contour, board_mask

def select_roi_and_get_color(frame):
    """
    Let the user select an ROI on the given frame.
    Compute the mean HSV color within the ROI and return computed lower/upper thresholds.
    """
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    if roi[2] == 0 or roi[3] == 0:
        print("ROI selection cancelled. Exiting.")
        exit()
    x, y, w, h = roi
    roi_frame = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    
    mean_hsv = cv2.mean(hsv_roi)[:3]
    mean_h = int(mean_hsv[0])
    mean_s = int(mean_hsv[1])
    mean_v = int(mean_hsv[2])
    
    tol_h = 15
    tol_s = 20
    tol_v = 10
    
    computed_lower = np.array([max(mean_h - tol_h, 0), max(mean_s - tol_s, 0), max(mean_v - tol_v, 0)])
    computed_upper = np.array([min(mean_h + tol_h, 179), min(mean_s + tol_s, 255), min(mean_v + tol_v, 255)])
    
    cv2.destroyWindow("Select ROI")
    print("Computed HSV color (mean):", (mean_h, mean_s, mean_v))
    print("Computed thresholds: lower =", computed_lower, "upper =", computed_upper)
    return computed_lower, computed_upper

def find_largest_red_cluster(red_mask):
    """
    Given a binary mask (red_mask), find the contour corresponding to the largest red cluster.
    """
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest

def fit_stick_line(contour):
    """
    Compute a rotated bounding box (using cv2.minAreaRect) for the contour,
    then extract the stick line as the line connecting the midpoints of the two shortest sides.
    The line is extended by 100 pixels.
    Returns two endpoints (pt1, pt2) as integers.
    """
    if contour is None or len(contour) == 0:
        return None, None

    # Compute the minimum area rectangle.
    rect = cv2.minAreaRect(contour)
    # Obtain the 4 corner points of the rotated rectangle.
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Compute each side's length and its endpoints.
    sides = []
    for i in range(4):
        ptA = box[i]
        ptB = box[(i+1) % 4]
        length = np.linalg.norm(ptA - ptB)
        sides.append((length, ptA, ptB))
    
    # Sort sides by length (smallest first).
    sides.sort(key=lambda x: x[0])
    # Take the two shortest sides.
    short_side_1 = sides[0]
    short_side_2 = sides[1]
    
    # Compute midpoints of these two short sides.
    middle_1 = (short_side_1[1] + short_side_1[2]) / 2
    middle_2 = (short_side_2[1] + short_side_2[2]) / 2

    # The base stick line is the line connecting these two midpoints.
    vector = middle_2 - middle_1
    norm = np.linalg.norm(vector)
    if norm == 0:
        return tuple(middle_1.astype(int)), tuple(middle_2.astype(int))
    vector = vector / norm

    # Extend the line from middle_2 by 100 pixels.
    extension = 100
    new_point = middle_2 + vector * extension

    pt1 = tuple(middle_1.astype(int))
    pt2 = tuple(new_point.astype(int))
    return pt1, pt2

def main():
    cap = cv2.VideoCapture(0)  # Change index if needed
    if not cap.isOpened():
        print("Error opening camera")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        return

    board_contour, board_mask = detect_board(frame, debug=False)
    if board_contour is None:
        print("Board not detected.")
        return

    computed_lower, computed_upper = select_roi_and_get_color(frame)

    # Use static thresholds for stick detection.
    # (Static range: Hue: 5-45, Saturation: 100-170, Value: 80-110)
    static_lower = np.array([5, 100, 80])
    static_upper = np.array([45, 170, 110])
    print("Using static thresholds: lower =", static_lower, "upper =", static_upper)

    print("Starting stick detection. Press ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, static_lower, static_upper)
        
        # Limit the red mask to the board region.
        red_mask_board = cv2.bitwise_and(red_mask, red_mask, mask=board_mask)
        
        largest_red_cluster = find_largest_red_cluster(red_mask_board)
        
        if largest_red_cluster is not None:
            pt1, pt2 = fit_stick_line(largest_red_cluster)
            if pt1 is not None and pt2 is not None:
                # Draw the blue stick line directly.
                cv2.line(frame, pt1, pt2, (255, 0, 0), 3)
                # Optionally, draw the rotated bounding box (in green) for debugging.
                box = cv2.boxPoints(cv2.minAreaRect(largest_red_cluster))
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        
        cv2.imshow("Stick Detection", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()