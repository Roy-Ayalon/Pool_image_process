import cv2
import numpy as np

def extract_table_edges_approx(contour, epsilon_ratio=0.05):
    """
    Approximates the given contour to a polygon and tries to extract 4 corners.
    
    :param contour: The contour (numpy array of shape [N,1,2]) outlining the table.
    :param epsilon_ratio: Fraction of the contour perimeter used as the approximation threshold.
    :return: A list of 4 lines, where each line is ((x1,y1),(x2,y2)), or None if not found.
    """
    # 1. Compute the perimeter of the contour
    perimeter = cv2.arcLength(contour, True)
    
    # 2. Approximate the polygon; increase epsilon_ratio if you still get extra corners
    epsilon = epsilon_ratio * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) != 4:
        print(f"Warning: Approximation did not yield 4 corners. Found {len(approx)} corners.")
        return None
    
    # 3. Convert the 4 corners into 4 lines
    lines = []
    for i in range(4):
        pt1 = tuple(approx[i][0])
        pt2 = tuple(approx[(i + 1) % 4][0])
        lines.append((pt1, pt2))
    
    return lines

def extract_table_edges_min_area(contour):
    """
    Uses cv2.minAreaRect to find a rotated bounding rectangle for the table contour.
    
    :param contour: The contour (numpy array of shape [N,1,2]) outlining the table.
    :return: A list of 4 lines, where each line is ((x1,y1),(x2,y2)).
    """
    # 1. Get the rotated rectangle that bounds the contour
    rect = cv2.minAreaRect(contour)  
    # rect contains ((center_x, center_y), (width, height), angle)
    
    # 2. Convert it to 4 corner points
    box_points = cv2.boxPoints(rect)  # shape (4,2)
    box_points = np.int0(box_points)  # convert to integer coords
    
    # 3. Form the 4 lines
    lines = []
    for i in range(4):
        pt1 = tuple(box_points[i])
        pt2 = tuple(box_points[(i + 1) % 4])
        lines.append((pt1, pt2))
    
    return lines

def draw_lines(image, lines, color=(0, 0, 255), thickness=2):
    """
    Draws each line in 'lines' onto 'image'.
    
    :param image: The BGR image on which to draw.
    :param lines: List of lines, where each line is ((x1,y1),(x2,y2)).
    :param color: BGR color for drawing.
    :param thickness: Line thickness.
    """
    if lines is None:
        return
    for (x1, y1), (x2, y2) in lines:
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)