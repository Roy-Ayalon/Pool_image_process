import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def merge_lines_with_extension(lines, axis, tolerance, height):
    """
    Merge lines and extend vertical lines to cover the entire height.
    
    Args:
        lines (list): List of lines, each represented as [(x1, y1, x2, y2)].
        axis (str): 'x' for vertical lines, 'y' for horizontal lines.
        tolerance (int): Distance threshold for clustering nearby lines.
        height (int): Height of the frame to extend vertical lines.
    
    Returns:
        list: Merged lines [(x1, y1, x2, y2), ...].
    """
    if not lines:
        return []

    # Extract line midpoints based on axis
    midpoints = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if axis == 'x':  # Vertical lines
            x_mid = (x1 + x2) // 2
            midpoints.append([x_mid, 0])  # Use x-coordinate for clustering
        elif axis == 'y':  # Horizontal lines
            y_mid = (y1 + y2) // 2
            midpoints.append([0, y_mid])  # Use y-coordinate for clustering

    # Perform clustering using DBSCAN
    clustering = DBSCAN(eps=tolerance, min_samples=1).fit(midpoints)
    unique_clusters = np.unique(clustering.labels_)

    merged_lines = []
    for cluster in unique_clusters:
        # Gather lines in this cluster
        cluster_lines = [lines[i] for i in range(len(lines)) if clustering.labels_[i] == cluster]
        
        # Average the coordinates to merge the lines
        if axis == 'x':  # Merge vertical lines
            x_avg = np.mean([(line[0][0] + line[0][2]) / 2 for line in cluster_lines])
            y_min = 160  # Extend to the top of the frame
            y_max = height-160  # Extend to the bottom of the frame
            merged_lines.append((int(x_avg), int(y_min), int(x_avg), int(y_max)))
        elif axis == 'y':  # Merge horizontal lines
            y_avg = np.mean([(line[0][1] + line[0][3]) / 2 for line in cluster_lines])
            x_min = min([line[0][0] for line in cluster_lines])
            x_max = max([line[0][2] for line in cluster_lines])
            merged_lines.append((int(x_min), int(y_avg), int(x_max), int(y_avg)))

    # Sort and keep only 2 lines for this axis (closest extremes)
    merged_lines = sorted(merged_lines, key=lambda l: l[0] if axis == 'x' else l[1])[:2]
    return merged_lines

def get_4_lines(frame):
    """
    Obtain a binary mask of the inner table region from the input frame,
    detect boundary lines using Hough Transform, and draw them on the frame.
    """
    img_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    _, A, _ = cv2.split(img_lab)
    A_blurred = cv2.GaussianBlur(A, (5, 5), 0)
    _, binary_mask = cv2.threshold(A_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plt.imshow(binary_mask, cmap='gray')
    plt.title("Binary Mask")
    plt.axis('off')
    plt.show()

    edges = cv2.Canny(binary_mask, threshold1=100, threshold2=200)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    plt.figure(figsize=(6, 4))
    plt.imshow(dilated_edges, cmap='gray')
    plt.title("Dilated Edges")
    plt.axis('off')
    plt.show()

    # Use HoughLinesP on dilated_edges now
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, threshold=200,
                            minLineLength=60, maxLineGap=10)
    if lines is None:
        print("No lines detected.")
        return frame  # or handle appropriately

    # Separate into vertical and horizontal lines
    vertical_lines = [line for line in lines if abs(line[0][0] - line[0][2]) < 50]  # x-coordinates similar
    horizontal_lines = [line for line in lines if abs(line[0][1] - line[0][3]) < 50]  # y-coordinates similar

    height, width = frame.shape[:2]

    # Merge lines to get exactly 2 vertical and 2 horizontal lines
    merged_verticals = merge_lines_with_extension(vertical_lines, axis='x', tolerance=50, height=height)
    merged_horizontals = merge_lines_with_extension(horizontal_lines, axis='y', tolerance=50, height=height)

    # i want to make mask of these 4 lines
    mask = np.zeros_like(binary_mask)
    for line in merged_verticals + merged_horizontals:
        x1, y1, x2, y2 = line
        cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
        
    return merged_verticals, merged_horizontals, mask

    # Draw the merged lines on the frame
    for line in merged_verticals + merged_horizontals:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display final result using matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 4))
    plt.title("Inner Table Boundary Lines")
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.show()
    
    return frame

