import cv2
import numpy as np

def detecet_holes(image, contour_polygon):
    """
    Detect holes in a table using the provided contour polygon.

    Parameters:
        image (ndarray): The input image (BGR format).
        contour_polygon (ndarray): A contour of the polygon (Nx1x2 array).

    Returns:
        contours (list): List of detected contours representing holes.
    """

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
    def subtract_shapes_mask(color_img, polygon_contour, rotated_rectangle):
        """
        Subtract the inner polygon from the rotated rectangle and debug the masks.
        """
        # Create blank masks
        polygon_mask = np.zeros(color_img.shape[:2], dtype=np.uint8)
        rect_mask = np.zeros(color_img.shape[:2], dtype=np.uint8)

        # Fill the polygon contour on the mask
        cv2.fillPoly(polygon_mask, [polygon_contour], 255)
        
        # Fill the rotated rectangle on a separate mask
        cv2.fillPoly(rect_mask, [rotated_rectangle], 255)

        # Debug shapes of masks
        print(f"Polygon mask shape: {polygon_mask.shape}")
        print(f"Rectangle mask shape: {rect_mask.shape}")

        # Subtract polygon mask from rectangle mask
        subtraction_result = cv2.subtract(rect_mask, polygon_mask)

        # Highlight the subtraction result on the original image
        result_img = color_img.copy()
        result_img[subtraction_result == 255] = [255, 0, 0]  # Highlight in blue

        return result_img, subtraction_result

    def detect_and_remove_long_lines(binary_mask, rho=1, theta=np.pi/180, threshold=100,
                                    min_line_length=100, max_line_gap=10, length_threshold=150):
        """
        Detect and remove long straight lines from a binary mask using the Hough Line Transform.
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

    # Calculate the rotated rectangle from the input contour_polygon
    rotated_rectangle = calculate_rotated_rectangle(contour_polygon)

    # Subtract the shapes to get a binary mask
    result_img, subtraction_result = subtract_shapes_mask(image, contour_polygon, rotated_rectangle)

    # Ensure binary mask is valid
    if subtraction_result is None or subtraction_result.size == 0:
        raise ValueError("Failed to generate binary mask. Check the inputs.")

    # Detect and remove long lines from the binary mask
    cleaned_mask, long_lines_mask, long_lines = detect_and_remove_long_lines(
        subtraction_result, rho=1, theta=np.pi/180, threshold=100,
        min_line_length=50, max_line_gap=10, length_threshold=150
    )

    # Find and draw contours on the original image
    contoured_image, binary, contours = find_and_draw_contours(image, cleaned_mask)

    return contours

# # Load an image
