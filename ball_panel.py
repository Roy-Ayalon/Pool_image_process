import cv2
import numpy as np

# Define the HSV ranges as provided.
COLOR_RANGES = {
    "yellow":   ((16,  147,  224),  (36, 167, 244)),  # new
    "brown":    ((0,   157,  150),  (40, 170, 175)),  # new
    "blue":     ((80,  130,  160),  (115, 220, 210)), # new
    "red":      ((0,   255,  255),  (0,   255,  255)),  # new
    "orange":   ((0,  160,  230),  (40, 210, 270)),  # NOT GOOD
    "green":    ((60,  130,  130),  (110, 230, 200)),  # new
    "purple":   ((90, 70,  100),  (150, 170, 180)), # new
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
    Creates a panel image showing ball icons by color in fixed positions.
    
    Fixed layout:
      - For each of the 7 colors (yellow, blue, red, purple, orange, green, brown),
        two slots are reserved.
      - For black, one slot is reserved.
      - White is omitted entirely.
    
    For each color:
      - If there are 0 balls, nothing is drawn in the reserved slot positions.
      - If there is 1 ball, only the left slot is filled.
      - If there are 2 balls, both slots are filled.
    
    Parameters:
        balls_info (list): Each element is a tuple (x, y, r, label, number).
        frame_width (int): Width of the panel (matches main frame width).
        panel_height (int): Height of the panel.
    
    Returns:
        panel (ndarray): An image with the drawn ball icons.
    """
    # Create a blank (white) panel.
    panel = 255 * np.ones((panel_height, frame_width, 3), dtype=np.uint8)
    
    # Define the layout order and the number of slots per color.
    # Colors are listed in the desired order.
    layout_order = [
        ("yellow", 2),
        ("blue", 2),
        ("red", 2),
        ("green", 2),
        ("brown", 2),
        ("black", 1)
    ]
    total_slots = sum(slots for _, slots in layout_order)  # should be 15
    
    # Count how many balls we have for each color (ignoring white and any other color).
    ball_counts = {color: 0 for color, _ in layout_order}
    for (_, _, _, label, _) in balls_info:
        color = label.lower()
        if color in ball_counts:
            ball_counts[color] += 1
    
    # Compute positions for each slot across the width.
    spacing = frame_width / (total_slots + 1)
    ball_radius = 20

    # Set up font parameters (smaller so the text fits inside the circle).
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Global slot counter to compute x positions.
    slot_counter = 0

    # Iterate through each color in the layout order.
    for color, slots in layout_order:
        # For each slot reserved for this color:
        for slot_index in range(slots):
            # Compute the center position for this slot.
            center_x = int(spacing * (slot_counter + 1))
            center_y = panel_height // 2
            slot_counter += 1

            # Determine if a ball should be drawn in this slot.
            # If one ball is available, fill only the left slot (slot_index 0).
            # If two balls are available, fill both slots.
            count = ball_counts[color]
            if count > slot_index:
                # Draw a filled circle for the ball.
                bgr_color = ball_color_mapping.get(color, (128, 128, 128))
                cv2.circle(panel, (center_x, center_y), ball_radius, bgr_color, -1)

                # Draw the color label inside the ball.
                text = color.capitalize()
                text_size, _ = cv2.getTextSize(text, font_face, font_scale, font_thickness)
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2 + 25
                cv2.putText(panel, text, (text_x, text_y), font_face, font_scale, (0, 0, 0), font_thickness)
            # Else: leave the slot blank (so the reserved position remains, but no ball is drawn).

    return panel