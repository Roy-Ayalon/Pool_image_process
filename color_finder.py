import cv2
import numpy as np

# Global variables for drawing
drawing = False
circle_center = (0, 0)
circle_radius = 0

def draw_circle(event, x, y, flags, param):
    """
    Callback function to draw a circle on mouse drag.
    """
    global drawing, circle_center, circle_radius

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        circle_center = (x, y)
        circle_radius = 0

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        circle_radius = int(np.sqrt((x - circle_center[0])**2 + (y - circle_center[1])**2))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        circle_radius = int(np.sqrt((x - circle_center[0])**2 + (y - circle_center[1])**2))

def get_average_hsv(frame, circle_center, circle_radius):
    """
    Computes the average HSV values inside the drawn circle.
    """
    if circle_radius == 0:
        return None  # No circle drawn

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the circle
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, circle_center, circle_radius, 255, -1)

    # Extract HSV values inside the circle
    hsv_values = hsv[mask == 255]

    # Compute average HSV
    if len(hsv_values) > 0:
        avg_h = int(np.mean(hsv_values[:, 0]))
        avg_s = int(np.mean(hsv_values[:, 1]))
        avg_v = int(np.mean(hsv_values[:, 2]))
        return avg_h, avg_s, avg_v
    else:
        return None  # No valid pixels in the mask

# Open camera
cap = cv2.VideoCapture(0)
cv2.namedWindow("HSV Color Detector")
cv2.setMouseCallback("HSV Color Detector", draw_circle)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if the camera fails

    # Draw the current circle
    if circle_radius > 0:
        cv2.circle(frame, circle_center, circle_radius, (0, 255, 0), 2)

    # Get HSV values inside the circle
    avg_hsv = get_average_hsv(frame, circle_center, circle_radius)
    if avg_hsv:
        avg_h, avg_s, avg_v = avg_hsv
        text = f"H: {avg_h}, S: {avg_s}, V: {avg_v}"
        cv2.putText(frame, text, (circle_center[0] - 40, circle_center[1] - circle_radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("HSV Color Detector", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
