import cv2
from detect_balls import detect_pool_balls
from detect_board import detect_board
from detect_holes import detecet_holes
from detect_stick import detect_stick

def capture_and_process_frame(cap):
    """Capture a single frame, apply ball detection, and return the processed frame."""
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        return None

    _, _, _, balls_contour = detect_pool_balls(frame)
    board_contour = detect_board(frame)
    holes_contours = detecet_holes(frame, board_contour)
    line = detect_stick(frame)

    # Draw the detected objects on the frame
    if board_contour is not None:
        cv2.drawContours(frame, [board_contour], -1, (255, 255, 0), thickness=2)
    if holes_contours:
        cv2.drawContours(frame, holes_contours, -1, (0, 255, 255), thickness=2)
    #if balls_contour:
        #cv2.drawContours(frame, balls_contour, -1, (0, 255, 0), thickness=2)
    # Assuming detect_stick returns a tuple/list of two points, each being (x, y)
    if line and isinstance(line, (list, tuple)) and len(line) == 2:
        pt1, pt2 = line
        # Ensure pt1 and pt2 are sequences of numbers with length 2
        if (hasattr(pt1, "__len__") and len(pt1) == 2 and 
            hasattr(pt2, "__len__") and len(pt2) == 2):
            cv2.line(frame, pt1, pt2, (0, 0, 255), thickness=3)
        else:
            print("Line points are not in the correct format:", line)
    else:
        print("No valid line detected or line is not in expected format.")

    return frame

def display_live_video(cap):
    """Continuously capture, process, and display video frames."""
    while True:
        processed_frame = capture_and_process_frame(cap)
        if processed_frame is None:
            break  # If frame capture fails, exit the loop

        cv2.imshow('Live Video', processed_frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):  # Wait for 33 ms and check for 'q' to quit
            break
    cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    try:
        display_live_video(cap)
    finally:
        cap.release()  # Ensure the camera is released properly

if __name__ == '__main__':
    main()