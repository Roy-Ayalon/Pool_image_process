import cv2
import numpy as np
from detect_4_lines_boundry import get_4_lines

def plot_trajectory(frame, line, holes_contour, board_lines, board_contour, balls_info):
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
            return
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

    # Helper function to draw text on the frame
    def draw_text_on_frame(text, color=(0, 255, 0)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = 50
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

    def plot_ball(frame, white_center, next_ball_center, next_ball_radius, stick_line, board_contour, dash_length=20, color=(255, 255, 0), thickness=2):
        """
        Predict and plot the trajectory of a white ball hitting another ball.

        Parameters:
            frame (ndarray): The input frame to draw on.
            white_center (tuple): The center (x, y) of the white ball.
            next_ball_center (tuple): The center (x, y) of the next ball to be hit.
            next_ball_radius (int): The radius of the next ball.
            stick_line (list): Two points defining the stick line [(x1, y1), (x2, y2)].
            board_contour (ndarray): Contour of the board.
            dash_length (int): Length of each dash in the dashed line.
            color (tuple): Color of the dashed line in BGR format.
            thickness (int): Thickness of the dashed line.

        Returns:
            None
        """

        def find_intersection_with_circle(center, radius, line_point1, line_point2):
            """
            Find the intersection of a line with a circle.

            Parameters:
                center (tuple): Center of the circle (x, y).
                radius (float): Radius of the circle.
                line_point1 (tuple): First point on the line.
                line_point2 (tuple): Second point on the line.

            Returns:
                tuple: Intersection point (x, y) or None if no intersection.
            """
            cx, cy = center
            x1, y1 = line_point1
            x2, y2 = line_point2

            # Line direction vector
            dx, dy = x2 - x1, y2 - y1

            # Quadratic coefficients
            a = dx**2 + dy**2
            b = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
            c = (x1 - cx)**2 + (y1 - cy)**2 - radius**2

            # Solve quadratic equation
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                return None  # No intersection

            # Find the two intersection points
            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)

            # Select the closest valid intersection point
            intersection1 = (x1 + t1 * dx, y1 + t1 * dy)
            intersection2 = (x1 + t2 * dx, y1 + t2 * dy)

            # Return the closer intersection point to the white ball center
            return intersection1 if np.linalg.norm(np.array(intersection1) - np.array(white_center)) < np.linalg.norm(np.array(intersection2) - np.array(white_center)) else intersection2

        def draw_dashed_line(start_point, slope, board_contour, color, dash_length, thickness):
            """
            Draw a dashed line with a given slope starting from a point.

            Parameters:
                start_point (tuple): The starting point (x, y) of the dashed line.
                slope (float): The slope of the line.
                board_contour (ndarray): Contour of the board.
                color (tuple): Color of the dashed line in BGR format.
                dash_length (int): Length of each dash in the dashed line.
                thickness (int): Thickness of the dashed line.

            Returns:
                None
            """
            x, y = start_point
            dx = 1
            dy = slope * dx
            norm = np.sqrt(dx**2 + dy**2)
            dx /= norm
            dy /= norm

            current_x, current_y = x, y

            for i in range(10000):  # Arbitrarily large limit
                next_x = current_x + dash_length * dx
                next_y = current_y + dash_length * dy

                # Check if next point is within board contour
                if cv2.pointPolygonTest(board_contour, (next_x, next_y), measureDist=False) < 0:
                    break

                # Draw dash
                dash_start = (int(current_x), int(current_y))
                dash_end = (int(next_x), int(next_y))
                cv2.line(frame, dash_start, dash_end, color, thickness)

                # Leave a gap
                current_x = next_x + dash_length * dx
                current_y = next_y + dash_length * dy

        # Step 1: Find the slope of the stick line
        (x1, y1), (x2, y2) = stick_line
        stick_slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')

        # Step 2: Define the white line starting from the white ball center with the stick slope
        white_line_point1 = white_center
        white_line_point2 = (white_center[0] + 1, white_center[1] + stick_slope) if stick_slope != float('inf') else (white_center[0], white_center[1] + 1)

        # Step 3: Find the hitting point (intersection with the next ball's circle)
        hitting_point = find_intersection_with_circle(next_ball_center, next_ball_radius, white_line_point1, white_line_point2)
        if hitting_point is None:
            print("No intersection with the next ball.")
            return

        # Step 4: Find the slope of the line from the hitting point to the center of the next ball
        next_ball_slope = (next_ball_center[1] - hitting_point[1]) / (next_ball_center[0] - hitting_point[0]) if hitting_point[0] != next_ball_center[0] else float('inf')

        # Step 5: Draw a dashed line starting at the center of the next ball with the calculated slope
        draw_dashed_line(next_ball_center, next_ball_slope, board_contour, color, dash_length, thickness)

        # Draw the hitting point for reference
        cv2.circle(frame, (int(hitting_point[0]), int(hitting_point[1])), 5, (0, 0, 255), -1)  # Red dot at hitting point


    def calculate_angle_between_lines(line1_slope, line2_slope):
        """
        Calculate the angle between two lines given their slopes.

        Parameters:
            line1_slope (float): Slope of the first line.
            line2_slope (float): Slope of the second line.

        Returns:
            float: Angle in radians between the two lines.
        """
        if line1_slope == float('inf') or line2_slope == float('inf'):  # Handle vertical lines
            return np.pi / 2
        tan_theta = abs((line2_slope - line1_slope) / (1 + line1_slope * line2_slope))
        return np.arctan(tan_theta)

    def plot_board(frame, stick_line, white_center, board_line, board_contours, dash_length=20, color=(255, 255, 0), thickness=2):
        """
        Plot the dashed reflected line based on the stick, board, and white ball.

        Parameters:
            frame (ndarray): The input frame to draw on.
            stick_line (list): Two points defining the stick line [(x1, y1), (x2, y2)].
            white_center (tuple): The center (x, y) of the white ball.
            board_line (list): Two points defining the board line [(x3, y3), (x4, y4)].
            board_contours (ndarray): Contour of the board.
            dash_length (int): Length of each dash in the dashed line.
            color (tuple): Color of the dashed line in BGR format.
            thickness (int): Thickness of the dashed line.

        Returns:
            None
        """
        # Step 1: Find the slope of the stick
        (x1, y1), (x2, y2) = stick_line
        stick_slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Vertical line handling

        # Step 2: Find the slope of the white line
        white_slope = stick_slope

        # Step 3: Find the intersection point of the white line with the board line
        (x3, y3), (x4, y4) = board_line
        board_slope = (y4 - y3) / (x4 - x3) if x4 != x3 else float('inf')  # Vertical line handling

        # Line equations: y = mx + c
        white_intercept = white_center[1] - white_slope * white_center[0] if white_slope != float('inf') else None
        board_intercept = y3 - board_slope * x3 if board_slope != float('inf') else None

        if white_slope == float('inf'):  # White line is vertical
            x_intersect = white_center[0]
            y_intersect = board_slope * x_intersect + board_intercept
        elif board_slope == float('inf'):  # Board line is vertical
            x_intersect = x3
            y_intersect = white_slope * x_intersect + white_intercept
        else:  # General case
            x_intersect = (board_intercept - white_intercept) / (white_slope - board_slope)
            y_intersect = white_slope * x_intersect + white_intercept

        intersection_point = (int(x_intersect), int(y_intersect))

        # Step 4: Calculate the angle between the white line and the perpendicular to the board line
        board_perpendicular_slope = -1 / board_slope if board_slope != float('inf') else 0  # Perpendicular slope
        angle = calculate_angle_between_lines(white_slope, board_perpendicular_slope)

        # Step 5: Plot dashed line in the mirrored direction
        mirrored_angle = -angle  # Mirror the angle
        dx = np.cos(mirrored_angle)
        dy = np.sin(mirrored_angle)

        # Extend the dashed line from the intersection point
        current_x, current_y = intersection_point
        total_dashes = 1000  # Limit to avoid infinite loops

        for i in range(total_dashes):
            next_x = current_x + dash_length * dx
            next_y = current_y + dash_length * dy

            # Check if the point is inside the board contours
            if cv2.pointPolygonTest(board_contours, (next_x, next_y), measureDist=False) < 0:
                break  # Exit if outside the board

            # Draw the dash
            dash_start = (int(current_x), int(current_y))
            dash_end = (int(next_x), int(next_y))
            cv2.line(frame, dash_start, dash_end, color, thickness)

            # Advance the current position, leaving a gap
            current_x = next_x + dash_length * dx
            current_y = next_y + dash_length * dy

    def plot_white(image, line, center, board_contour, dash_length=10, color=(0, 255, 0), thickness=2):
        """
        - line of the stick
        - center of the white ball
        """

        # Extract line points and compute slope
        (x1, y1), (x2, y2) = line
        dx = x2 - x1
        dy = y2 - y1

        # Normalize direction vector
        direction = np.array([dx, dy], dtype=float)
        direction /= np.linalg.norm(direction)

        # Start point of the dashed line
        start_x, start_y = center

        # Extend the line until it intersects with the board contour
        extended_line = []
        for i in range(1, 10000):  # Arbitrarily large number to extend the line
            current_x = start_x + i * direction[0]
            current_y = start_y + i * direction[1]
            extended_line.append([current_x, current_y])

            # Check if this point intersects the board contour
            if cv2.pointPolygonTest(board_contour, (current_x, current_y), measureDist=False) < 0:
                break  # Exit if outside the board contour

        # End point of the dashed line
        end_x, end_y = extended_line[-1]

        # Draw the dashed line
        current_x, current_y = center
        total_distance = int(np.linalg.norm([end_x - current_x, end_y - current_y]))
        for step in range(0, total_distance, 2 * dash_length):
            # Calculate start and end of the dash segment
            segment_start = (
                int(current_x + step * direction[0]),
                int(current_y + step * direction[1]),
            )
            segment_end = (
                int(current_x + (step + dash_length) * direction[0]),
                int(current_y + (step + dash_length) * direction[1]),
            )
            # Draw the segment
            cv2.line(image, segment_start, segment_end, color, thickness)

    # Check if it's okay to shoot
    okay_to_shoot = okay_to_shoot(frame, line, balls_info)
    # if not okay_to_shoot:
    #     draw_text_on_frame("Not okay to shoot.", color=(0, 0, 255))  # Red text
    #     return

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

    # Check intersection with balls - need to change this part to check for intersection for each ball
    for ball in balls_info:
        x, y, r, _, _ = ball
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

    # Check intersection with the board - need to change this part to check for intersection with each edge of the board
    _, _, board_mask = get_4_lines(frame)

    intersection = cv2.bitwise_and(line_mask, board_mask)
    if np.any(intersection):
        draw_text_on_frame("Intersects board", color=(255, 0, 0))  # Blue text
        return

    # If no intersections are found
    draw_text_on_frame("No intersections", color=(0, 255, 0))  # Green text

def compute_trajectory(white_center, white_radius, cue_line):
    """
    Computes the trajectory direction of the white ball given the cue stick line.
    
    Parameters:
        white_center (tuple): (x, y) coordinates of the white ball center.
        white_radius (int): Radius of the white ball.
        cue_line (list): Two points [(x1, y1), (x2, y2)] along the cue stick.
        
    Returns:
        dir_unit (ndarray): The normalized direction vector of the white ball (pointing away from the cue tip).
        contact_point (tuple): The point on the white ball's circumference where the cue strikes.
    """
    C = np.array(white_center, dtype=float)
    p0 = np.array(cue_line[0], dtype=float)
    p1 = np.array(cue_line[1], dtype=float)
    
    # Choose the cue tip as the endpoint farthest from the white ball center.
    if np.linalg.norm(p0 - C) > np.linalg.norm(p1 - C):
        cue_tip = p0
    else:
        cue_tip = p1
    
    # The ball is struck so that its impulse is in the direction from the cue tip to the white ball center.
    dir_vec = C - cue_tip
    norm = np.linalg.norm(dir_vec)
    if norm == 0:
        # Fallback to the cue stick direction if the white center equals the tip (degenerate case)
        dir_unit = (p1 - p0) / np.linalg.norm(p1 - p0)
    else:
        dir_unit = dir_vec / norm

    # Compute the contact point on the white ball's circumference.
    contact_point = C + white_radius * dir_unit

    return dir_unit, tuple(contact_point.astype(int))


def okay_to_shoot(frame, line, white_ball, balls_info):
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
            return
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
        # Calculate direction vector
        dx, dy = x2 - x1, y2 - y1
        line_vector = np.array([dx, dy], dtype=float)

        # Check if the vector has zero length (to avoid division by zero)
        norm = np.linalg.norm(line_vector)
        if norm == 0:
            print("Error: Zero-length line vector, cannot normalize.")
            return False  # or handle differently

        # Normalize the vector
        line_vector /= norm

        # Calculate extended points
        extended_start = (int(x1 - line_vector[0] * line_length), int(y1 - line_vector[1] * line_length))
        extended_end = (int(x2 + line_vector[0] * line_length), int(y2 + line_vector[1] * line_length))

        # Draw the extended line on the mask
        cv2.line(line_mask, extended_start, extended_end, 255, thickness=5)

        # Create a blank mask for the balls
        balls_mask = np.zeros((height, width), dtype=np.uint8)

        # Draw each ball as a filled circle on the balls mask
        x, y, r = white_ball
        cv2.circle(balls_mask, (int(x), int(y)), int(r), 255, thickness=cv2.FILLED)

        # Check for intersection between the line and balls masks
        intersection = cv2.bitwise_and(line_mask, balls_mask)
        has_intersection = np.any(intersection)

        return has_intersection

def extend_line(P1, P2, length=1000):
    """
    Extend a line segment from P1 -> P2 by 'length' pixels.
    """
    P1, P2 = np.array(P1, dtype=float), np.array(P2, dtype=float)
    direction = P2 - P1
    direction = direction / np.linalg.norm(direction)  # Normalize
    extended_P2 = P1 + direction * length
    return tuple(extended_P2.astype(int))

def find_first_intersecting_ball(contact_point, dir_unit, balls_info, board_contour, step_size=5, max_length=1000):
    """
    Finds the first ball that intersects the white ball's computed trajectory.

    Parameters:
        contact_point (tuple): The starting point (x, y) of the trajectory.
        dir_unit (ndarray): The normalized direction vector of the white ball.
        balls_info (list): List of detected balls [(x, y, r, label, number)].
        board_contour (ndarray): Contour of the board (to stop the trajectory if needed).
        step_size (int): Distance step for iterating along the trajectory.
        max_length (int): Maximum trajectory length to check.

    Returns:
        intersecting_ball (tuple or None): The first ball that intersects the trajectory, or None if no intersection.
        intersection_point (tuple or None): The point where the trajectory first intersects a ball.
    """
    current_point = np.array(contact_point, dtype=float)
    total_length = 0

    while total_length < max_length:
        # Advance along the trajectory
        next_point = current_point + step_size * dir_unit
        
        # Stop if outside board contour
        if cv2.pointPolygonTest(board_contour, (next_point[0], next_point[1]), False) < 0:
            return None, None  # No intersection before leaving the board
        
        # Check for intersection with any ball
        for ball in balls_info:
            bx, by, br, label, number = ball  # Ball center (bx, by), radius (br), and label
            
            # Compute distance between trajectory point and ball center
            distance = np.linalg.norm(next_point - np.array([bx, by]))

            if distance <= br:  # Collision detected
                return ball, tuple(next_point.astype(int))

        # Move forward
        current_point = next_point
        total_length += step_size

    return None, None  # No ball was hit within max_length

def compute_next_trajectory(intersecting_ball, intersection_point, dir_unit):
    """
    Computes the trajectory directions after the white ball collides with another ball.

    Parameters:
        intersecting_ball (tuple): The ball that was hit (x, y, r, label, number).
        intersection_point (tuple): The point where the white ball first makes contact.
        dir_unit (ndarray): The current direction vector of the white ball.

    Returns:
        new_white_dir (ndarray): The new direction of the white ball after impact.
        target_ball_dir (ndarray): The direction of the target ball after impact.
    """
    bx, by, br, label, number = intersecting_ball
    B = np.array([bx, by], dtype=float)  # Center of the ball that was hit
    P = np.array(intersection_point, dtype=float)  # The impact point

    # Compute the direction for the target ball (it moves away from the impact point)
    target_ball_dir = B - P
    target_ball_dir = target_ball_dir / np.linalg.norm(target_ball_dir)  # Normalize

    # Compute the new white ball direction after impact (reflection model)
    # The white ball is deflected perpendicular to the impact direction
    normal = target_ball_dir  # Normal is the same as the target ball direction
    new_white_dir = dir_unit - 2 * np.dot(dir_unit, normal) * normal  # Reflection formula

    return new_white_dir, target_ball_dir

def check_hole_intersection(contact_point, dir_unit, holes_mask, max_length=100000, line_thickness=5):
    """
    Creates a mask for the white ball's trajectory (from contact_point in the direction of dir_unit)
    and checks if it intersects with the holes mask.

    Parameters:
        contact_point (tuple): (x, y) coordinates where the trajectory begins.
        dir_unit (ndarray): Normalized 2D direction vector.
        holes_mask (ndarray): Binary mask (e.g., 0 or 255) for the holes.
        max_length (int): Maximum length (in pixels) to extend the trajectory line.
        line_thickness (int): Thickness with which to draw the trajectory line in the mask.

    Returns:
        hit (bool): True if the trajectory line intersects a hole, False otherwise.
        line_mask (ndarray): The mask image with the drawn trajectory.
        intersection (ndarray): The result of the bitwise AND between line_mask and holes_mask.
    """
    # Ensure holes_mask is single-channel.
    if len(holes_mask.shape) == 3 and holes_mask.shape[2] > 1:
        holes_mask_gray = cv2.cvtColor(holes_mask, cv2.COLOR_BGR2GRAY)
    else:
        holes_mask_gray = holes_mask

    # Create a blank mask (single-channel) matching holes_mask_gray.
    line_mask = np.zeros_like(holes_mask_gray, dtype=np.uint8)
    
    # Compute an endpoint along the trajectory.
    contact_arr = np.array(contact_point, dtype=float)
    endpoint = contact_arr + max_length * np.array(dir_unit, dtype=float)
    endpoint = tuple(np.round(endpoint).astype(int))
    
    # Draw the trajectory line on the blank mask.
    cv2.line(line_mask, (int(contact_point[0]), int(contact_point[1])), endpoint, 255, thickness=line_thickness)
    
    # Now compute the intersection of the line mask with the holes mask.
    intersection = cv2.bitwise_and(line_mask, holes_mask_gray)
    
    # Check if there is any intersection (non-zero pixel)
    hit = cv2.countNonZero(intersection) > 0

    # Visualize the line mask and holes mask
    # cv2.imshow("Line Mask", line_mask)
    # cv2.imshow("Holes Mask", holes_mask_gray)
    # cv2.imshow("inter", intersection)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return hit, line_mask, intersection


def get_reflection_direction(dir_unit, board_edges_mask, intersection_point, window_size=10):
    """
    Given the incident direction (dir_unit), board edges mask, and the intersection point,
    compute the reflection direction based on a local estimation of the wall's normal.
    
    Parameters:
        dir_unit (ndarray): Normalized 2D incident direction vector.
        board_edges_mask (ndarray): Binary mask for the board edges.
        intersection_point (tuple): (x, y) coordinates of the collision point.
        window_size (int): Half-size of the window to extract local edge pixels.
        
    Returns:
        reflection (ndarray or None): Normalized reflection direction vector,
                                      or None if a local edge cannot be estimated.
    """
    x, y = intersection_point
    h, w = board_edges_mask.shape[:2]
    
    # Define the region of interest (ROI) around the intersection point.
    x1 = max(0, x - window_size)
    x2 = min(w, x + window_size)
    y1 = max(0, y - window_size)
    y2 = min(h, y + window_size)
    
    # Extract the patch from the board edges mask.
    patch = board_edges_mask[y1:y2, x1:x2]
    
    # Find nonzero points (i.e., edge pixels) in the patch.
    pts = cv2.findNonZero(patch)
    if pts is None or len(pts) < 2:
        # Not enough points to determine a line.
        return None

    # Reshape points to (N, 2) and convert to absolute coordinates.
    pts = pts.reshape(-1, 2)
    pts[:, 0] += x1
    pts[:, 1] += y1

    # Fit a line to these points using cv2.fitLine.
    # The output is in the form (vx, vy, x0, y0) where (vx, vy) is the unit vector of the line.
    line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    vx, vy, _, _ = line

    # The wall's tangent is (vx, vy). Its normal can be taken as (-vy, vx) (or the opposite).
    normal = np.array([-vy, vx])
    normal = normal / np.linalg.norm(normal)
    
    # Compute the reflection direction using the formula:
    # reflected = v - 2 * (v dot n) * n
    reflection = dir_unit - 2 * (np.dot(dir_unit, normal)) * normal
    reflection = reflection / np.linalg.norm(reflection)
    
    return reflection

def check_board_edge_intersection(contact_point, dir_unit, board_edges_mask, max_length=100000, line_thickness=5):
    """
    Checks if the white ball's trajectory (starting from contact_point along dir_unit)
    intersects the board edges mask and returns the first intersection point.

    Parameters:
        contact_point (tuple): (x, y) coordinates where the trajectory begins.
        dir_unit (ndarray): Normalized 2D direction vector.
        board_edges_mask (ndarray): Binary mask (e.g., 0 or 255) representing the board edges.
        max_length (int): Maximum length (in pixels) to extend the trajectory line.
        line_thickness (int): Thickness with which to draw the trajectory line in the mask.

    Returns:
        hit (bool): True if the trajectory line intersects a board edge, False otherwise.
        line_mask (ndarray): The mask image with the drawn trajectory.
        intersection (ndarray): The result of the bitwise AND between line_mask and board_edges_mask.
        intersection_point (tuple or None): The (x, y) coordinates of the intersection point,
                                             or None if no intersection was found.
    """
    # Ensure board_edges_mask is single-channel.
    if len(board_edges_mask.shape) == 3 and board_edges_mask.shape[2] > 1:
        board_edges_gray = cv2.cvtColor(board_edges_mask, cv2.COLOR_BGR2GRAY)
    else:
        board_edges_gray = board_edges_mask

    # Create a blank mask matching board_edges_gray.
    line_mask = np.zeros_like(board_edges_gray, dtype=np.uint8)

    # Compute an endpoint along the trajectory.
    contact_arr = np.array(contact_point, dtype=float)
    endpoint = contact_arr + max_length * np.array(dir_unit, dtype=float)
    endpoint = tuple(np.round(endpoint).astype(int))

    # Draw the trajectory line on the blank mask.
    cv2.line(line_mask, (int(contact_point[0]), int(contact_point[1])), endpoint, 255, thickness=line_thickness)

    # Compute the intersection of the trajectory with the board edges.
    intersection = cv2.bitwise_and(line_mask, board_edges_gray)
    hit = cv2.countNonZero(intersection) > 0

    intersection_point = None
    reflection_direction = None
    if hit:
        # Get all the non-zero points (intersection pixels).
        pts = cv2.findNonZero(intersection)
        if pts is not None:
            pts = pts.reshape(-1, 2)  # Convert shape from (N,1,2) to (N,2)
            # Compute the squared distances from the contact point.
            distances = np.sum((pts - np.array(contact_point))**2, axis=1)
            # The closest point is considered the first intersection point.
            min_index = np.argmin(distances)
            intersection_point = tuple(pts[min_index])

            # Compute the reflection direction from the estimated local wall normal.
            reflection_direction = get_reflection_direction(dir_unit, board_edges_gray, intersection_point)

    return hit, line_mask, intersection, intersection_point, reflection_direction