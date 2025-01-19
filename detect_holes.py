import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = '/Users/mikitatarjitzky/Documents/DIP_project/WhatsApp Image 2024-12-29 at 09.40.28.jpeg'
image = cv2.imread(image_path)

mask_polygon_path = '/Users/mikitatarjitzky/Documents/DIP_project/mask.png'
mask_polygon = cv2.imread(mask_polygon_path, cv2.IMREAD_GRAYSCALE)

def detecet_holes(image, mask_polygon):
    # Extract polygon points from the mask
    def extract_polygon_points(mask):
        """
        Extract the largest polygon (contour) points from a binary mask.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in the mask.")
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour

    # Calculate the rotated rectangle enclosing the polygon
    def calculate_rotated_rectangle(polygon_points):
        """
        Calculate the minimum-area rotated rectangle for the polygon points.
        """
        rect = cv2.minAreaRect(polygon_points)
        rotated_rectangle = cv2.boxPoints(rect)  # Convert to 4 corner points
        rotated_rectangle = np.int32(rotated_rectangle)  # Convert to integer
        return rotated_rectangle

    # Subtract the shapes
    def subtract_shapes_mask(color_img, mask_polygon, rotated_rectangle):
        """
        Subtract the inner contour from the rotated rectangle and debug the masks.
        """
        mask_polygon = cv2.resize(mask_polygon, (color_img.shape[1], color_img.shape[0]))

        # Ensure the masks have the same size as the input image
        if mask_polygon.shape != color_img.shape[:2]:
            raise ValueError("Mask polygon size does not match the input image size.")
        
        # Create blank mask for the rotated rectangle
        rect_mask = np.zeros(color_img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(rect_mask, [rotated_rectangle], 255)

        # Debug shapes of masks
        print(f"Contour mask shape: {mask_polygon.shape}")
        print(f"Rectangle mask shape: {rect_mask.shape}")

        # Subtract contour mask from rectangle mask
        subtraction_result = cv2.subtract(rect_mask, mask_polygon)

        # Highlight the subtraction result on the original image
        result_img = color_img.copy()
        result_img[subtraction_result == 255] = [255, 0, 0]  # Highlight in blue

        return result_img, subtraction_result

    ## if i have a mask of the table i can find the holes in the table by subtracting the mask from the image

    def detect_and_remove_long_lines(binary_mask, rho=1, theta=np.pi/180, threshold=100,
                                    min_line_length=100, max_line_gap=10, length_threshold=150):
        """
        Detect and remove long straight lines from a binary mask using the Hough Line Transform.

        Parameters:
            binary_mask (ndarray): Input binary mask.
            rho (float): Distance resolution of the accumulator in pixels.
            theta (float): Angle resolution of the accumulator in radians.
            threshold (int): Accumulator threshold for detecting lines.
            min_line_length (int): Minimum length of a line to be detected.
            max_line_gap (int): Maximum allowed gap between points on the same line.
            length_threshold (int): Minimum length of a line to consider it "long."

        Returns:
            cleaned_mask (ndarray): Binary mask with long lines removed.
            long_lines_mask (ndarray): Binary mask of detected long lines.
            long_lines (list): List of detected long lines [(x1, y1, x2, y2)].
        """
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(binary_mask, rho, theta, threshold,
                                minLineLength=min_line_length, maxLineGap=max_line_gap)

        # Create a blank mask for the detected long lines
        long_lines_mask = np.zeros_like(binary_mask)

        long_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate the length of the line
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                # Keep only lines longer than the length threshold
                if length > length_threshold:
                    cv2.line(long_lines_mask, (x1, y1), (x2, y2), 255, thickness=3)
                    long_lines.append((x1, y1, x2, y2))

        # Remove long lines from the original mask
        cleaned_mask = cv2.subtract(binary_mask, long_lines_mask)

        return cleaned_mask, long_lines_mask, long_lines

    def find_and_draw_contours(original_image, cleaned_mask, min_area=2000, max_area=10000):
        """
        Find contours in the cleaned mask, filter them by area, and overlay them on the original image.

        Parameters:
            original_image (ndarray): The original input image (BGR format).
            cleaned_mask (ndarray): The binary mask with lines removed.
            min_area (int): Minimum area of contours to keep.
            max_area (int): Maximum area of contours to keep.

        Returns:
            filtered_image (ndarray): The original image with filtered contours drawn.
            filtered_contours (list): List of filtered contours.
        """
        # Find contours in the cleaned mask
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area
        filtered_contours = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]

        # Create a copy of the original image to draw contours
        filtered_image = original_image.copy()

        # Draw the filtered contours on the original image
        cv2.drawContours(filtered_image, filtered_contours, -1, (0, 255, 0), 2)  # Green contours

        # Create a binary mask with the inside of filtered contours filled
        binary_mask = np.zeros(cleaned_mask.shape, dtype=np.uint8)
        cv2.drawContours(binary_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

        return filtered_image, binary_mask, filtered_contours

    # Extract polygon points
    polygon_points = extract_polygon_points(mask_polygon)

    # Calculate the rotated rectangle
    rotated_rectangle = calculate_rotated_rectangle(polygon_points)

    # Subtract the shapes
    result_img, subtraction_result = subtract_shapes_mask(image, mask_polygon, rotated_rectangle)

    binary_mask = subtraction_result

    # Ensure binary mask is valid
    if binary_mask is None:
        raise ValueError("Failed to load binary mask. Check the file path.")

    # Detect and remove long lines from the binary mask
    cleaned_mask, long_lines_mask, long_lines = detect_and_remove_long_lines(
        binary_mask, rho=1, theta=np.pi/180, threshold=100,
        min_line_length=50, max_line_gap=10, length_threshold=150
    )

    # Find and draw contours on the original image
    contoured_image, binary, contours = find_and_draw_contours(image, cleaned_mask)

    return contours

