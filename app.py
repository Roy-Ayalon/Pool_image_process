import cv2
from detect_balls import detect_pool_balls
from detect_board import detect_board
from detect_holes import detecet_holes
from detect_stick import detect_stick
from table_start import table_start
from trajectory import plot_trajectory

board_contour = None

def capture_and_process_frame(cap, mask_for_stick):
    """Capture a single frame, apply ball detection, and return the processed frame."""
    global board_contour
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        return None
    # detect board only once
    if board_contour is None:
        board_contour = detect_board(frame)
    # Unpack all returned values correctly; ensure your detect_pool_balls signature matches this unpacking.
    annotated, balls_info, ball_mask, balls_contour = detect_pool_balls(frame, board_contour)
    holes_contours = detecet_holes(frame, board_contour)
    line = detect_stick(frame, mask_for_stick)
    #plot_trajectory(frame, line, holes_contours, board_contour, balls_info)

    # show type of balls_contour
    # print(type(balls_contour[0]))

    
               

    # Draw the detected objects on the frame
    if board_contour is not None:
        cv2.drawContours(frame, [board_contour], -1, (0, 255, 0), thickness=2)
    if holes_contours:
        cv2.drawContours(frame, holes_contours, -1, (0, 255, 0), thickness=2)
    # Check if balls_contour is valid and non-empty
    if balls_contour is not None and len(balls_contour) > 0:
        for ball_info in balls_info:
            x, y, r, ball_label, _ = ball_info
            cv2.circle(frame, (x, y), r, (255, 0, 0), 2)
            cv2.putText(frame, ball_label, (x - r, y - r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


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
    ret, frame = cap.read()
    mask_for_stick = table_start(frame, 200)
    while True:
        processed_frame = capture_and_process_frame(cap, mask_for_stick)
        if processed_frame is None:
            break  # If frame capture fails, exit the loop

        cv2.imshow('Live Video', processed_frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):  # Wait for 33 ms and check for 'q' to quit
            break
    cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    try:
        display_live_video(cap)
    finally:
        cap.release()  # Ensure the camera is released properly

if __name__ == '__main__':
    main()