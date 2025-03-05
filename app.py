import cv2
import numpy as np

from detect_balls import detect_pool_balls, detect_white_ball
from detect_board import detect_board
from detect_holes import detecet_holes
from detect_stick import detect_stick
from trajectory import okay_to_shoot, compute_trajectory, extend_line
import matplotlib.pyplot as plt
from ball_panel import create_balls_panel
from is_moving import check_white_ball_movement  # Updated function signature

# Global persistent dictionary for remaining balls.
remaining_balls = {}

# Global point counter.
points = 0

# Global variable to store the previous white ball center.
previous_white_ball_center = None

def main():
    global remaining_balls, points, previous_white_ball_center

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Modes: idle, board_captured, mask (balls detected), game.
    mode = "idle"
    board_contour = None
    binary_image = None  # Board mask from detect_board()
    binary_mask = None   # Also obtained when detecting the board

    print("Press 'b' to detect board, 'a' to detect balls on table, 's' to start game, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Preprocess frame with a blur.
        frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        display_frame = frame.copy()

        if mode == "idle":
            cv2.putText(display_frame, "Idle Mode: Press 'b' to detect board", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Live Video", display_frame)

        elif mode == "board_captured":
            if board_contour is not None:
                cv2.drawContours(display_frame, [board_contour], -1, (0, 0, 255), 2)
            cv2.putText(display_frame, "Board Captured: Press 'a' to detect balls", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Live Video", display_frame)

        elif mode == "mask":
            annotated, balls_info, ball_mask, balls_contour, binary_balls = detect_pool_balls(frame_blurred, board_contour)
            
            if board_contour is not None:
                cv2.drawContours(display_frame, [board_contour], -1, (0, 0, 255), 2)
            
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
            # Re-detect balls (or track them) in game mode.
            annotated, balls_info, ball_mask, balls_contour, binary_balls = detect_pool_balls(frame_blurred, board_contour)
            
            if board_contour is not None:
                cv2.drawContours(display_frame, [board_contour], -1, (0, 0, 255), 2)
            
            if balls_contour is not None and len(balls_contour) > 0:
                for ball_info in balls_info:
                    x, y, r, ball_label, _ = ball_info
                    cv2.circle(display_frame, (x, y), r, (0, 0, 255), 2)
                    cv2.putText(display_frame, ball_label, (x - r, y - r - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # --- White Ball Movement Check ---
            _, white_ball = detect_white_ball(display_frame, board_contour)
            white_ball_moving = False
            if white_ball is not None:
                white_center = (white_ball[0], white_ball[1])
                white_ball_moving = check_white_ball_movement(white_center, previous_white_ball_center, frame, threshold=20)
                # Update the previous center after checking.
                previous_white_ball_center = white_center

            if white_ball_moving:
                cv2.putText(display_frame, "White ball moving - stick detection skipped", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # --- Stick Detection Code (only when white ball is stationary) ---
                res_img, start_point, end_point = detect_stick(frame, binary_mask)
                if start_point is not None and end_point is not None:
                    cv2.line(display_frame, start_point, end_point, (255, 0, 0), 3)

                    # --- Physics and Trajectory Computation ---
                    line = (start_point, end_point)
                    # Re-detect white ball to ensure an up-to-date position.
                    _, white_ball = detect_white_ball(display_frame, board_contour)
                    if white_ball is not None:
                        okay = okay_to_shoot(display_frame, line, white_ball, balls_info)
                        overlay_text = "True" if okay else "False"
                        text_color = (0, 255, 0) if okay else (0, 0, 255)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.5
                        thickness = 3
                        text_size = cv2.getTextSize(overlay_text, font, font_scale, thickness)[0]
                        text_x = (frame.shape[1] - text_size[0]) // 2
                        text_y = (frame.shape[0] + text_size[1]) // 2
                        cv2.putText(display_frame, overlay_text, (text_x, text_y), font, font_scale, text_color, thickness)

                        if okay:
                            white_x, white_y, white_radius = white_ball
                            white_center = (white_x, white_y)
                            dir_unit, contact_point = compute_trajectory(white_center, white_radius, line)
                            print(dir_unit, contact_point)

                            step = 5  # Small step size for trajectory advancement
                            current_point = np.array(contact_point, dtype=float)
                            total_length = 0
                            max_length = 1000
                            while total_length < max_length:
                                next_point = current_point + step * dir_unit
                                if cv2.pointPolygonTest(board_contour, (next_point[0], next_point[1]), False) < 0:
                                    break
                                current_point = next_point
                                total_length += step

                            cv2.line(display_frame,
                                     (int(contact_point[0]), int(contact_point[1])),
                                     (int(current_point[0]), int(current_point[1])),
                                     color=(255, 255, 0), thickness=2)

            cv2.putText(display_frame, "Game Mode: Press 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # --- Update remaining balls and score ---
            valid_colors = ["yellow", "blue", "red", "purple", "orange", "green", "brown", "black"]
            colored_valid = ["yellow", "blue", "red", "purple", "orange", "green", "brown"]
            max_counts = {"yellow": 2, "blue": 2, "red": 2, "purple": 2, "orange": 2, "green": 2, "brown": 2, "black": 1}

            current_detected = {color: [] for color in valid_colors}
            for ball in balls_info:
                x, y, r, label, number = ball
                color_label = label.lower()
                if color_label == "white" or color_label not in valid_colors:
                    continue
                current_detected[color_label].append(ball)

            for color in valid_colors:
                if len(current_detected[color]) > max_counts[color]:
                    current_detected[color] = current_detected[color][:max_counts[color]]

            if not any(remaining_balls.values()):
                remaining_balls = {color: current_detected[color] for color in valid_colors}
                points = 0
            else:
                prev_total = sum(len(remaining_balls[color]) for color in colored_valid)
                current_total = sum(len(current_detected[color]) for color in colored_valid)
                delta = prev_total - current_total
                points += delta
                for color in valid_colors:
                    remaining_balls[color] = current_detected[color]

            cv2.putText(display_frame, f"Score: {points}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
            board_contour, binary_image, binary_mask, board_perimeter_mask, holes_mask = detect_board(frame)
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