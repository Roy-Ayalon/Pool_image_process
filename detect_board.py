import cv2
import numpy as np
import matplotlib.pyplot as plt
from detect_balls import detect_pool_balls


def detect_board(frame, debug=False):
    """
    Detect the board contour from a frame using LAB color space and adaptive thresholding.
    Returns:
        Largest contour and a binary mask of the board.
    """
    # Convert the image to LAB color space and normalize the A channel
    lab_image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab_image)
    A_normalized = cv2.normalize(A, None, 0, 255, cv2.NORM_MINMAX)

    if debug:
        cv2.imshow("Normalized A Channel", A_normalized)
        cv2.waitKey(0)

    # Compute histogram and determine adaptive threshold
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

    # Find contours and extract the largest one (assumed to be the table)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        if debug:
            print("No contours found.")
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a black image to draw the detected board
    black_image = np.zeros_like(frame)
    cv2.drawContours(black_image, [largest_contour], -1, (0, 255, 0), 2)

    def rgb_to_cmyk(image):
        """
        Convert an RGB image to CMYK format
        :param image: Input image in RGB
        :return: CMYK image
        """
        # Normalize the image to [0, 1]
        image = image / 255.0
    
        # Get RGB channels
        R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    
        # Compute CMY channels
        C = 1 - R
        M = 1 - G
        Y = 1 - B
    
        # Compute the K channel
        K = np.minimum(C, np.minimum(M, Y))
    
        # Avoid division by zero for the non-K channels
        C = (C - K) / (1 - K + 1e-5)
        M = (M - K) / (1 - K + 1e-5)
        Y = (Y - K) / (1 - K + 1e-5)
    
        # Stack the CMYK channels
        cmyk = np.dstack((C, M, Y, K))
    
        return cmyk

    
    # Convert RGB to CMYK
    img_cmyk = rgb_to_cmyk(frame)
    c, m, y, k = cv2.split(img_cmyk)
    new = np.where(k < 0.7, 1, 0)
    G = black_image[:,:,1]
    no_holes = G*new



    #make no holes as cv2 image
    no_holes = np.where(no_holes > 0, 255, 0).astype(np.uint8)
    #cv2.imshow("no_holes", no_holes)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # apply hough transform to detect lines, and then extract the 4 lines of the table
    #edges = cv2.Canny(no_holes, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(no_holes, 1, np.pi / 180, 200)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(black_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    no_holes = np.where(black_image > 0, 255, 0).astype(np.uint8)



    # split no holes to red and green colors
    red = no_holes[:,:,2]
    green = no_holes[:,:,1]

    # show the red and green channels
    #cv2.imshow("red", red)
    #cv2.imshow("green", green)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Separate lines into vertical and horizontal groups based on theta
    vertical_lines = []
    horizontal_lines = []
    for line in lines:
        rho, theta = line[0]
        # Nearly vertical: theta near 0 or pi
        if theta < np.pi / 4 or theta > 3 * np.pi / 4:
            vertical_lines.append((rho, theta))
        else:
            horizontal_lines.append((rho, theta))

    # For vertical lines, compute x-intercepts (for y = 0) and split into left/right using the median
    vertical_lines_with_x = [(rho, theta, rho/np.cos(theta)) for (rho, theta) in vertical_lines if np.cos(theta) != 0]
    if vertical_lines_with_x:
        x_intercepts = [x for (_, _, x) in vertical_lines_with_x]
        median_x = np.median(x_intercepts)
        left_lines = [(rho, theta) for (rho, theta, x) in vertical_lines_with_x if x < median_x]
        right_lines = [(rho, theta) for (rho, theta, x) in vertical_lines_with_x if x >= median_x]
    else:
        left_lines, right_lines = [], []

    # For horizontal lines, compute y-intercepts (for x = 0) and split into top/bottom using the median
    horizontal_lines_with_y = [(rho, theta, rho/np.sin(theta)) for (rho, theta) in horizontal_lines if np.sin(theta) != 0]
    if horizontal_lines_with_y:
        y_intercepts = [y for (_, _, y) in horizontal_lines_with_y]
        median_y = np.median(y_intercepts)
        top_lines = [(rho, theta) for (rho, theta, y) in horizontal_lines_with_y if y < median_y]
        bottom_lines = [(rho, theta) for (rho, theta, y) in horizontal_lines_with_y if y >= median_y]
    else:
        top_lines, bottom_lines = [], []

    # make mask for the table perimeter
    binary_mask_perimeter = np.zeros_like(frame)
    # draw only 4 lines
    for line in left_lines[:1] + right_lines[:1] + top_lines[:1] + bottom_lines[:1]:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(binary_mask_perimeter, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    #cv2.imshow("binary_mask_perimeter", binary_mask_perimeter)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # take green , and find 4 corners of the table
    green = np.where(green > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in green segmentation.")
    # take 4 largest contours, and make hole mask
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
    hole_mask = np.zeros_like(frame)
    cv2.drawContours(hole_mask, largest_contours, -1, (255, 255, 255), cv2.FILLED)
    #cv2.imshow("hole_mask", hole_mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    

    


    



    return largest_contour, black_image, binary_mask, binary_mask_perimeter, hole_mask

import cv2
import numpy as np

def detect_table_by_grey(frame, debug=False):
    """
    Detects the pool table outer boundary by segmenting the grey lines.
    Uses HSV color segmentation to extract low-saturation (grey) regions,
    then finds the largest contour, approximates it to a polygon, and finally 
    computes the four edge segments.
    
    Parameters:
        frame (numpy.ndarray): Input BGR image.
        debug (bool): If True, displays intermediate results.
    
    Returns:
        table_contour (numpy.ndarray): The detected contour for the table.
        table_lines (list of tuples): List of four edge segments as ((x1,y1),(x2,y2)).
        grey_mask (numpy.ndarray): The binary mask from grey segmentation.
    """
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for grey colors: low saturation and moderate value.
    lower_grey = np.array([0, 80, 150])
    upper_grey = np.array([50, 150, 200])
    
    # Create mask that isolates grey regions (the painted table lines)
    grey_mask = cv2.inRange(hsv, lower_grey, upper_grey)
    
    # Clean up noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    grey_mask = cv2.morphologyEx(grey_mask, cv2.MORPH_CLOSE, kernel)
    grey_mask = cv2.morphologyEx(grey_mask, cv2.MORPH_OPEN, kernel)
    
    if debug:
        cv2.imshow("Grey Mask", grey_mask)
        cv2.waitKey(0)
    
    # Find contours in the grey mask
    contours, _ = cv2.findContours(grey_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in grey segmentation.")
    
    # Assume the largest contour is the table boundary
    table_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to a polygon. Ideally, we want 4 corners.
    epsilon = 0.02 * cv2.arcLength(table_contour, True)
    approx = cv2.approxPolyDP(table_contour, epsilon, True)
    
    approx = approx.reshape(-1, 2)
    
    # Order the points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = approx.sum(axis=1)
    rect[0] = approx[np.argmin(s)]   # top-left
    rect[2] = approx[np.argmax(s)]     # bottom-right
    diff = np.diff(approx, axis=1)
    rect[1] = approx[np.argmin(diff)]  # top-right
    rect[3] = approx[np.argmax(diff)]  # bottom-left
    
    # Create the four edge segments from the rectangle
    top_edge    = (tuple(rect[0]), tuple(rect[1]))
    right_edge  = (tuple(rect[1]), tuple(rect[2]))
    bottom_edge = (tuple(rect[2]), tuple(rect[3]))
    left_edge   = (tuple(rect[3]), tuple(rect[0]))
    
    table_lines = [top_edge, right_edge, bottom_edge, left_edge]
    
    if debug:
        # Draw detected contour and lines on a copy of the frame.
        debug_frame = frame.copy()
        cv2.drawContours(debug_frame, [approx.astype(int)], -1, (0, 255, 0), 2)
        for line in table_lines:
            cv2.line(debug_frame, tuple(map(int, line[0])), tuple(map(int, line[1])), (0, 0, 255), 2)
        cv2.imshow("Detected Table Boundary", debug_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return table_contour, table_lines, grey_mask


def subtract_shapes_mask(frame, largest_contour):
    """
    Subtract the largest contour from the frame to isolate the table.
    Parameters:
        frame (numpy.ndarray): Input BGR image.
        largest_contour (numpy.ndarray): The largest contour to subtract.
    Returns:
        result_img (numpy.ndarray): The resulting image after subtraction.
        subtraction_result (numpy.ndarray): The binary mask of the subtraction result.
    """
    # Create a black mask with the same dimensions as the frame
    mask = np.zeros_like(frame)
    
    # Draw the largest contour filled with white color on the mask
    cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), cv2.FILLED)
    
    # Subtract the mask from the frame
    result_img = cv2.subtract(frame, mask)
    
    # Convert the subtraction result to a binary mask
    subtraction_result = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    _, subtraction_result = cv2.threshold(subtraction_result, 1, 255, cv2.THRESH_BINARY)
    
    return result_img, subtraction_result
