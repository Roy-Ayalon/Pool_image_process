import numpy as np

def check_white_ball_movement(white_center, frame, threshold=5):
    """
    Checks if the white ball is moving by comparing its current center with the previous center.

    Parameters:
        white_center (tuple): The current (x, y) center of the white ball.
        frame (ndarray): The current frame (not used for computation but available for further analysis if needed).
        threshold (float): Minimum displacement in pixels to consider the ball as moving.

    Returns:
        bool: True if the white ball is moving, False otherwise.
    """
    global previous_white_ball_center
    # If no previous position is recorded, assume the ball is stationary.
    if previous_white_ball_center is None:
        previous_white_ball_center = white_center
        return False

    # Compute the Euclidean distance between the current and previous centers.
    displacement = np.linalg.norm(np.array(white_center) - np.array(previous_white_ball_center))
    previous_white_ball_center = white_center  # Update the previous center for the next frame

    return displacement > threshold