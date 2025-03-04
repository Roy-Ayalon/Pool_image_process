import cv2
import numpy as np

from detect_tip import find_largest_red_cluster, fit_stick_line
from detect_balls import detect_pool_balls
from detect_board import detect_board
from detect_holes import detecet_holes
from detect_stick import detect_stick
from table_start import table_start
from trajectory import plot_trajectory
import matplotlib.pyplot as plt
from ball_panel import create_balls_panel
from four_lines import extract_table_edges_approx, extract_table_edges_min_area, draw_lines

# Global persistent dictionary for remaining balls.
# Keys are ball numbers; values are (x, y, r, label, number) tuples.
remaining_balls = {}

# Global point counter.
points = 0

def main():
    global remaining_balls, points  # Declare these as global

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
            
            # --- Update persistent remaining balls and score (by color only, ignoring black for scoring) ---
            valid_colors = ["yellow", "blue", "red", "purple", "orange", "green", "brown", "black"]
            # Only colored balls count toward the score.
            colored_valid = ["yellow", "blue", "red", "purple", "orange", "green", "brown"]
            max_counts = {"yellow": 2, "blue": 2, "red": 2, "purple": 2, "orange": 2, "green": 2, "brown": 2, "black": 1}

            # Build a dictionary of currently detected balls (ignoring the white ball).
            current_detected = {color: [] for color in valid_colors}
            for ball in balls_info:
                # Each ball: (x, y, r, label, number)
                x, y, r, label, number = ball
                color_label = label.lower()
                if color_label == "white" or color_label not in valid_colors:
                    continue
                current_detected[color_label].append(ball)

            # Limit each color to its maximum allowed count.
            for color in valid_colors:
                if len(current_detected[color]) > max_counts[color]:
                    current_detected[color] = current_detected[color][:max_counts[color]]

            # If remaining_balls has not been initialized (first frame), initialize it and set score to 0.
            if not any(remaining_balls.values()):
                remaining_balls = {color: current_detected[color] for color in valid_colors}
                points = 0
            else:
                # Compute total counts only for the colored balls.
                prev_total = sum(len(remaining_balls[color]) for color in colored_valid)
                current_total = sum(len(current_detected[color]) for color in colored_valid)
                delta = prev_total - current_total  # Positive if balls went missing (pocketed), negative if reappeared.
                points += delta

                # Update the persistent record.
                for color in valid_colors:
                    remaining_balls[color] = current_detected[color]

            # Compute total missing colored balls (expected total is 14).
            total_missing = sum(max_counts[color] - len(remaining_balls[color]) for color in colored_valid)


            # --- Check win/loss conditions ---
            # If all 14 colored balls are pocketed, display "YOU WON"
            if total_missing == 14:
                h, w = display_frame.shape[:2]
                cv2.putText(display_frame, "YOU WON", (w//2 - 150, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
                cv2.imshow("Live Video", display_frame)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('w'):
                    # Reset game state.
                    points = 0
                    remaining_balls = {color: [] for color in valid_colors}
                    mode = "game"
                    continue
                if key == ord('q'):
                    break

            # Else, if the black ball is missing (pocketed) but not all colored balls are pocketed, you lose.
            elif len(remaining_balls["black"]) == 0 and total_missing < 14:
                h, w = display_frame.shape[:2]
                cv2.putText(display_frame, "YOU LOSE", (w//2 - 150, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
                cv2.imshow("Live Video", display_frame)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('l'):
                    points = 0
                    remaining_balls = {color: [] for color in valid_colors}
                    mode = "game"
                    continue
                if key == ord('q'):
                    break

            # Display the current score on the frame.
            cv2.putText(display_frame, f"Score: {points}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # --- Create and display the balls panel ---
            # Combine all currently detected balls (from all valid colors) into a single list.
            present_balls = []
            for color in valid_colors:
                present_balls.extend(remaining_balls[color])
            if present_balls:
                h, w = display_frame.shape[:2]
                panel = create_balls_panel(present_balls, w, panel_height=100)
                combined_frame = np.vstack([display_frame, panel])
                cv2.imshow("Live Video", combined_frame)
            else:
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