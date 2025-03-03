import cv2
import numpy as np

from detect_tip import find_largest_red_cluster, fit_stick_line
from detect_balls import detect_pool_balls
from detect_board import detect_board
from detect_holes import detecet_holes
from detect_stick import detect_stick
from table_start import table_start
from trajectory import trajectory
import matplotlib.pyplot as plt
from detect_tip import stick
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Modes: idle, board_captured, mask (balls detected), game.
    mode = "idle"
    board_contour = None
    binary_image = None  # <-- This is our board mask from detect_board()

    print("Press 'b' to detect board, 'a' to detect balls on table, 's' to start game, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Preprocess frame with a blur
        frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        display_frame = frame.copy()

        if mode == "idle":
            cv2.putText(display_frame, "Idle Mode: Press 'b' to detect board", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Live Video", display_frame)

        elif mode == "board_captured":
            if board_contour is not None:
                # board_contour is just points we can draw to visualize the board
                cv2.drawContours(display_frame, [board_contour], -1, (0, 0, 255), 2)
            cv2.putText(display_frame, "Board Captured: Press 'a' to detect balls", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Live Video", display_frame)

        elif mode == "mask":
            # Detect balls and draw red circles on live frame.
            annotated, balls_info, ball_mask, balls_contour, binary_balls = detect_pool_balls(frame_blurred, board_contour)
            
            # Visualize board contour
            if board_contour is not None:
                cv2.drawContours(display_frame, [board_contour], -1, (0, 0, 255), 2)
            
            # Visualize balls
            if balls_contour is not None and len(balls_contour) > 0:
                for ball_info in balls_info:
                    x, y, r, ball_label, _ = ball_info
                    cv2.circle(display_frame, (x, y), r, (0, 0, 255), 2)
                    cv2.putText(display_frame, ball_label, (x - r, y - r - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(display_frame, "Balls Detected: Press 's' to start game", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Live Video", display_frame)

        elif mode == "game":
            # Re-detect balls (or track them) in game mode
            annotated, balls_info, ball_mask, balls_contour, binary_balls = detect_pool_balls(frame_blurred, board_contour)
            
            # Visualize board contour
            if board_contour is not None:
                cv2.drawContours(display_frame, [board_contour], -1, (0, 0, 255), 2)
            
            # Visualize balls
            if balls_contour is not None and len(balls_contour) > 0:
                for ball_info in balls_info:
                    x, y, r, ball_label, _ = ball_info
                    cv2.circle(display_frame, (x, y), r, (0, 0, 255), 2)
                    cv2.putText(display_frame, ball_label, (x - r, y - r - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # --- Tip Detection Code ---
            # Static HSV thresholds
            static_lower = np.array([5, 100, 80])
            static_upper = np.array([45, 170, 110])
            
            # Convert current frame to HSV and threshold
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            tip_mask = cv2.inRange(hsv, static_lower, static_upper)
            
            # Use the actual binary mask from detect_board(), not board_contour
            if binary_mask is not None:
                board_mask = binary_mask.copy()
                
                # Make sure shapes match
                if board_mask.shape != tip_mask.shape:
                    board_mask = cv2.resize(board_mask, (tip_mask.shape[1], tip_mask.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
                # Ensure uint8
                if board_mask.dtype != np.uint8:
                    board_mask = board_mask.astype(np.uint8)
                
                # Restrict the tip detection to the board region
                tip_mask_board = cv2.bitwise_and(tip_mask, tip_mask, mask=board_mask)
                
                # Find the largest red cluster
                largest_tip_cluster = find_largest_red_cluster(tip_mask_board)
                if largest_tip_cluster is not None:
                    tip_pt1, tip_pt2 = fit_stick_line(largest_tip_cluster)
                    if tip_pt1 is not None and tip_pt2 is not None:
                        cv2.line(display_frame, tip_pt1, tip_pt2, (255, 0, 0), 3)
            else:
                # If we have no board mask, we can't do tip_mask_board
                print("No board mask available; press 'b' to detect board first.")
            # --- End Tip Detection Code ---
            
            cv2.putText(display_frame, "Game Mode: Press 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Live Video", display_frame)

        key = cv2.waitKey(33) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            # detect_board should return (board_contour, binary_image)
            board_contour, binary_image, binary_mask = detect_board(frame)
            print("Board detected and saved.")
            mode = "board_captured"
        elif key == ord('a'):
            if mode == "board_captured":
                print("Detecting balls on table...")
                mode = "mask"
        elif key == ord('s'):
            if mode in ["mask", "board_captured"]:
                print("Game started!")
                mode = "game"

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()