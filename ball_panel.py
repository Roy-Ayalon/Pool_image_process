import cv2
import numpy as np

# Define the HSV ranges as provided.
COLOR_RANGES = {
    "yellow":   ((16,  147,  224),  (36, 167, 244)),  # new
    "brown":    ((0,   157,  150),  (40, 170, 175)),  # new
    "blue":     ((80,  130,  160),  (115, 220, 210)), # new
    "red":      ((115,   125,  210),  (179, 230, 255)),  # new
    "orange":   ((0,  160,  230),  (40, 210, 270)),  # NOT GOOD
    "green":    ((60,  130,  130),  (110, 230, 200)),  # new
 #   "purple":   ((90, 70,  100),  (150, 170, 180)), # new
    "white":    ((60,   0,    230),  (100, 20, 255)),  # hue ~0-180, sat ~0-50, val ~200-255
    "black":    ((30,   120,    0),    (85, 190, 50))    # new
}


def hsv_mean_to_bgr(lower, upper):
    """
    Computes the mean of lower and upper HSV bounds and converts it to BGR.
    
    Parameters:
        lower (tuple): Lower HSV bound.
        upper (tuple): Upper HSV bound.
    
    Returns:
        tuple: BGR color as a tuple of ints.
    """
    mean_h = (lower[0] + upper[0]) / 2.0
    mean_s = (lower[1] + upper[1]) / 2.0
    mean_v = (lower[2] + upper[2]) / 2.0
    mean_hsv = np.array([mean_h, mean_s, mean_v], dtype=np.float32)
    mean_hsv_uint8 = np.uint8([[mean_hsv]])
    mean_bgr = cv2.cvtColor(mean_hsv_uint8, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(x) for x in mean_bgr)

# Create the ball color mapping based on the mean HSV values.
ball_color_mapping = {}
for color, (lower, upper) in COLOR_RANGES.items():
    ball_color_mapping[color] = hsv_mean_to_bgr(lower, upper)

def create_balls_panel(balls_info, frame_width, panel_height=100):
    """
    Creates a panel image showing ball icons by color (ignoring ball numbers).
    - Up to 2 balls for each of the 7 main colors
    - Up to 1 black ball
    - White ball is omitted entirely.
    - Maximum 15 total.

    Parameters:
        balls_info (list): Each element is (x, y, r, label, number).
        frame_width (int): Width of the main frame so the panel matches.
        panel_height (int): Height of the panel.

    Returns:
        panel (ndarray): An image with drawn ball icons.
    """
    # Create a blank panel (black background)
    panel = 255 * np.ones((panel_height, frame_width, 3), dtype=np.uint8)
    
    # Define which colors we want to allow and how many of each
    valid_colors = ["yellow", "blue", "red", "purple", "orange", "green", "brown", "black"]
    max_counts = {
        "yellow": 2, "blue": 2, "red": 2, "purple": 2,
        "orange": 2, "green": 2, "brown": 2, "black": 1
    }
    color_count = {c: 0 for c in valid_colors}

    # Filter and limit to the desired colors/quantities
    selected_balls = []
    for x, y, r, label, number in balls_info:
        color_label = label.lower()
        # Skip white or any color not in our valid list
        if color_label not in valid_colors:
            continue
        # Skip if we already have the max for that color
        if color_count[color_label] < max_counts[color_label]:
            selected_balls.append((x, y, r, color_label))
            color_count[color_label] += 1

    # Now draw these selected balls on the panel
    num_balls = len(selected_balls)
    spacing = frame_width / (num_balls + 1) if num_balls > 0 else frame_width

    ball_radius = 20
    for i, (x, y, r, color_label) in enumerate(selected_balls):
        center_x = int(spacing * (i + 1))
        center_y = panel_height // 2

        # Get the BGR color, default to gray if unknown
        color = ball_color_mapping.get(color_label.lower(), (128, 128, 128))

        # Draw filled circle with the ball's color
        cv2.circle(panel, (center_x, center_y), ball_radius, color, -1)

        # Optionally label with the color name (ignoring the number)
        text = color_label.capitalize()
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2 + 25
        cv2.putText(panel, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return panel