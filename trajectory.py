import cv2
import numpy as np

def okay_to_shoot(frame, line, balls_info):
    """
    Determine if it is okay to shoot based on the line and ball positions, using balls_info.

    Parameters:
        frame (ndarray): The input image.
        line (list): A list of two points [(x1, y1), (x2, y2)] defining the stick.
        balls_info (list): List of tuples [(x, y, r), ...], where (x, y) is the center and r is the radius of each ball.

    Returns:
        bool: True if the line intersects with any ball, False otherwise.
    """

    # Validate input
    if line is None or len(line) != 2:
        raise ValueError("Line must be a list of two points [(x1, y1), (x2, y2)].")
    if not balls_info or not isinstance(balls_info, list):
        raise ValueError("balls_info must be a list of (x, y, r) tuples.")

    # Extract the start and end points of the line
    (x1, y1), (x2, y2) = line

    # Get the image dimensions
    height, width = frame.shape[:2]

    # Create a blank mask for the line
    line_mask = np.zeros((height, width), dtype=np.uint8)

    # Extend the line across the image
    line_length = max(width, height) * 2  # Large length to ensure extension
    dx, dy = x2 - x1, y2 - y1
    line_vector = np.array([dx, dy], dtype=float)
    line_vector /= np.linalg.norm(line_vector)  # Normalize

    # Calculate extended points
    extended_start = (int(x1 - line_vector[0] * line_length), int(y1 - line_vector[1] * line_length))
    extended_end = (int(x2 + line_vector[0] * line_length), int(y2 + line_vector[1] * line_length))

    # Draw the extended line on the mask
    cv2.line(line_mask, extended_start, extended_end, 255, thickness=5)

    # Create a blank mask for the balls
    balls_mask = np.zeros((height, width), dtype=np.uint8)

    # Draw each ball as a filled circle on the balls mask
    for ball in balls_info:
        x, y, r, _, _ = ball
        cv2.circle(balls_mask, (int(x), int(y)), int(r), 255, thickness=cv2.FILLED)

    # Check for intersection between the line and balls masks
    intersection = cv2.bitwise_and(line_mask, balls_mask)
    has_intersection = np.any(intersection)

    # Overlay the result on the frame
    overlay_text = "True" if has_intersection else "False"
    text_color = (0, 255, 0) if has_intersection else (0, 0, 255)  # Green for True, Red for False

    # Display the text in the center of the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    text_size = cv2.getTextSize(overlay_text, font, font_scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2

    cv2.putText(frame, overlay_text, (text_x, text_y), font, font_scale, text_color, thickness)

    return has_intersection

def plot_trajectory(frame, okay_to_shoot, line, holes_contour, board_contour, balls_info):
    """
    Plot the trajectory and determine the first object the line intersects.

    Parameters:
        frame (ndarray): The input image.
        okay_to_shoot (bool): Whether it is okay to shoot.
        line (list): A list of two points [(x1, y1), (x2, y2)] defining the stick.
        holes_contour (list): Contour of the holes on the board.
        board_contour (ndarray): Contour of the board without holes.
        balls_info (list): List of (x, y, r) tuples for the balls on the table.

    Returns:
        None
    """
    # Helper function to draw text on the frame
    def draw_text_on_frame(text, color=(0, 255, 0)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = 50
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

    # Check if it's okay to shoot
    if not okay_to_shoot:
        draw_text_on_frame("Not okay to shoot.", color=(0, 0, 255))  # Red text
        return

    # Extract the start and end points of the line
    (x1, y1), (x2, y2) = line

    # Get the image dimensions
    height, width = frame.shape[:2]

    # Extend the line across the image
    line_length = max(width, height) * 2  # Large length to ensure extension
    dx, dy = x2 - x1, y2 - y1
    line_vector = np.array([dx, dy], dtype=float)
    line_vector /= np.linalg.norm(line_vector)  # Normalize

    # Calculate extended points
    extended_start = (int(x1 - line_vector[0] * line_length), int(y1 - line_vector[1] * line_length))
    extended_end = (int(x2 + line_vector[0] * line_length), int(y2 + line_vector[1] * line_length))

    # Create a blank mask for the line
    line_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.line(line_mask, extended_start, extended_end, 255, thickness=5)

    # Check intersection with balls
    for ball in balls_info:
        x, y, r = ball
        # Create a mask for the ball
        ball_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(ball_mask, (int(x), int(y)), int(r), 255, thickness=cv2.FILLED)

        # Check for intersection
        intersection = cv2.bitwise_and(line_mask, ball_mask)
        if np.any(intersection):
            draw_text_on_frame(f"Intersects ball at ({x}, {y})", color=(255, 255, 0))  # Yellow text
            return

    # Check intersection with holes
    for hole_contour in holes_contour:
        # Create a mask for the hole
        hole_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(hole_mask, [hole_contour], -1, 255, thickness=cv2.FILLED)

        # Check for intersection
        intersection = cv2.bitwise_and(line_mask, hole_mask)
        if np.any(intersection):
            draw_text_on_frame("Intersects hole", color=(0, 255, 255))  # Cyan text
            return

    # Check intersection with the board
    board_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(board_mask, [board_contour], -1, 255, thickness=cv2.FILLED)

    intersection = cv2.bitwise_and(line_mask, board_mask)
    if np.any(intersection):
        draw_text_on_frame("Intersects board", color=(255, 0, 0))  # Blue text
        return

    # If no intersections are found
    draw_text_on_frame("No intersections", color=(0, 255, 0))  # Green text
