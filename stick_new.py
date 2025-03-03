import cv2
import numpy as np


def preprocess_frame_with_contour(frame, contour):
    """
    Preprocess the frame by creating a mask based on a given contour.

    :param frame: Input image frame.
    :param contour: The contour defining the region of interest.
    :return: Masked frame where only the area inside the contour is preserved.
    """
    # Create an empty mask the same size as the frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Fill the contour area with white (255) to create a mask
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to keep only the relevant area
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    return masked_frame, mask  # Return both the masked frame and the mask itself

def highlight_color(frame, contour, target_bgr=(49, 74, 82), tolerance=25):
    """
    Detects a certain color within a given frame (after applying a mask) and highlights it.

    :param frame: Input image frame.
    :param mask: Preprocessing mask to apply before color detection.
    :param target_bgr: The BGR color to detect.
    :param tolerance: Allowed variation in color detection.
    :return: Frame with detected color highlighted, and a mask showing detected areas.
    """
    # Preprocess the frame with the provided mask
    preprocessed_frame, _ = preprocess_frame_with_contour(frame, contour)

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2HSV)

    # Convert the target BGR color to HSV
    target_hsv = np.uint8([[list(target_bgr)]])  # Convert to a NumPy array
    target_hsv = cv2.cvtColor(target_hsv, cv2.COLOR_BGR2HSV)[0][0]  # Convert to HSV

    lower = target_hsv - 55
    upper = target_hsv + 55

    # Define lower and upper bounds for color detection
    lower_np = np.array(lower, dtype=np.uint8)
    upper_np = np.array(upper, dtype=np.uint8)

    # Create a mask to detect the color within the range
    color_mask = cv2.inRange(hsv_frame, lower_np, upper_np)

    # Create an output frame that starts as a copy of the original preprocessed frame
    highlighted_frame = preprocessed_frame.copy()

    # Change the detected color to a distinct color (e.g., red)
    highlight_color = np.array([0, 0, 255])  # Red in BGR
    highlighted_frame[color_mask != 0] = highlight_color  # Replace detected color with red

    return highlighted_frame, color_mask
