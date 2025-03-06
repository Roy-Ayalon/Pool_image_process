import cv2
import numpy as np

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