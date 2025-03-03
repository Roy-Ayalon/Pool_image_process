import cv2
import numpy as np

def find_largest_red_cluster(red_mask):
    """
    Given a binary mask (red_mask), find the contour corresponding to the largest red cluster.
    Returns:
        The contour with the largest area, or None if no contours are found.
    """
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def fit_stick_line(contour):
    """
    Compute a rotated bounding box (using cv2.minAreaRect) for the given contour,
    then extract the stick line as the line connecting the midpoints of the two
    shortest sides. Extend that line by 100 pixels.

    Returns:
        pt1, pt2 (as (x, y) integer tuples), or (None, None) if contour is empty.
    """
    if contour is None or len(contour) == 0:
        return None, None

    # Compute the minimum area rectangle for the contour.
    rect = cv2.minAreaRect(contour)
    # Obtain the 4 corner points.
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Compute each side's length and keep track of endpoints.
    sides = []
    for i in range(4):
        ptA = box[i]
        ptB = box[(i + 1) % 4]
        length = np.linalg.norm(ptA - ptB)
        sides.append((length, ptA, ptB))
    
    # Sort sides by length (smallest first) and take the two shortest sides.
    sides.sort(key=lambda x: x[0])
    short_side_1 = sides[0]
    short_side_2 = sides[1]
    
    # Midpoints of the two shortest sides.
    middle_1 = (short_side_1[1] + short_side_1[2]) / 2.0
    middle_2 = (short_side_2[1] + short_side_2[2]) / 2.0

    # The base stick line is the line connecting these two midpoints.
    vector = middle_2 - middle_1
    norm = np.linalg.norm(vector)
    if norm == 0:
        # Degenerate case: both midpoints are the same
        return tuple(middle_1.astype(np.int32)), tuple(middle_2.astype(np.int32))
    vector = vector / norm

    # Extend the line from the second midpoint by 100 pixels.
    extension = 100
    extended_point = middle_2 + vector * extension

    pt1 = tuple(middle_1.astype(np.int32))
    pt2 = tuple(extended_point.astype(np.int32))
    return pt1, pt2

def get_stick_tip_points(frame, board_mask):
    """
    Given a frame and a board_mask (binary image), this function:
      1) Applies fixed HSV thresholding (Hue: 5–45, Sat: 100–170, Val: 80–110) to extract the tip region.
      2) Ensures board_mask is the same size as tip_mask and is uint8.
      3) Masks the tip region by the board area (bitwise_and).
      4) Finds the largest red cluster in that masked region.
      5) Computes the stick line via minAreaRect (fit_stick_line).

    Returns:
        (pt1, pt2): endpoints of the stick tip line as integer tuples
        or (None, None) if no valid tip is found.
    """

    if board_mask is None:
        # No mask provided -> can't do masking
        return None, None

    # Define the static HSV thresholds for your tip.
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