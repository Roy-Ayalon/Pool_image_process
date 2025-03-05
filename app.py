import cv2
import numpy as np

from detect_balls import detect_pool_balls, detect_white_ball
from detect_board import detect_board
from detect_holes import detecet_holes
from detect_stick import detect_stick
from trajectory import okay_to_shoot, compute_trajectory, extend_line, find_first_intersecting_ball, compute_next_trajectory, check_board_collision
import matplotlib.pyplot as plt
from ball_panel import create_balls_panel


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

            # --- Stick Detection Code ---
            res_img, start_point, end_point = detect_stick(frame, binary_mask)
            if start_point is not None and end_point is not None:
                cv2.line(display_frame, start_point, end_point, (255, 0, 0), 3)

                # --- physics ---
                line = (start_point, end_point)
                _, white_ball = detect_white_ball(display_frame, board_contour)
                if white_ball is not None:
                    okay = okay_to_shoot(display_frame, line, white_ball, balls_info)
                    # Display the text in the center of the frame
                    overlay_text = "True" if okay else "False"
                    text_color = (0, 255, 0) if okay else (0, 0, 255)  # Green for True, Red for False
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    thickness = 3
                    text_size = cv2.getTextSize(overlay_text, font, font_scale, thickness)[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = (frame.shape[0] + text_size[1]) // 2
                    cv2.putText(display_frame, overlay_text, (text_x, text_y), font, font_scale, text_color, thickness)

                    if okay:
                        # --- white ball line ---
                        white_x, white_y, white_radius = white_ball
                        white_center = (white_x, white_y)

                        # Compute initial trajectory
                        dir_unit, contact_point = compute_trajectory(white_center, white_radius, line)

                        # --- White Ball First Trajectory ---
                        intersecting_ball, intersection_point = find_first_intersecting_ball(contact_point, dir_unit, balls_info, board_contour)

                        step = 5  # Small step size for trajectory advancement
                        current_point = np.array(contact_point, dtype=float)
                        total_length = 0
                        max_length = 1000

                        # Check if a colored ball is hit
                        if intersecting_ball:
                            trajectory_end = np.array(intersection_point, dtype=float)  # Stop at the ball
                        else:
                            # If no ball is hit, continue to the board contour or holes
                            while total_length < max_length:
                                next_point = current_point + step * dir_unit

                                # Check if the next point is still inside the board contour
                                if cv2.pointPolygonTest(board_contour, (next_point[0], next_point[1]), False) < 0:
                                    break  # Stop drawing when exiting the board

                                current_point = next_point
                                total_length += step

                            trajectory_end = current_point  # Stop at the board

                        # Check for board edge or hole collision for the white ball
                        trajectory_end, new_dir = check_board_collision(trajectory_end, dir_unit, board_edges_mask, holes_mask)

                        # Draw the white ball's trajectory (Yellow)
                        cv2.line(display_frame,
                                (int(contact_point[0]), int(trajectory_end[0])),
                                (int(trajectory_end[1]), int(trajectory_end[1])),
                                color=(255, 255, 0), thickness=4)

                        # If the white ball falls into a hole, stop further calculations
                        if new_dir is None:
                            print("White ball pocketed!")
                        else:
                            # If bounced off the board edge, continue with the new direction
                            current_point = np.array(trajectory_end, dtype=float)
                            total_length = 0

                            while total_length < max_length:
                                next_point = current_point + step * new_dir

                                # Stop if outside board contour
                                if cv2.pointPolygonTest(board_contour, (next_point[0], next_point[1]), False) < 0:
                                    break

                                current_point = next_point
                                total_length += step

                            # Draw the new white ball trajectory after bounce (Cyan)
                            cv2.line(display_frame,
                                    (int(trajectory_end[0]), int(trajectory_end[1])),
                                    (int(current_point[0]), int(current_point[1])),
                                    color=(0, 255, 255), thickness=4)

                        # --- If a ball is hit, compute its new trajectory ---
                        if intersecting_ball:
                            print(f"Ball {intersecting_ball[3]} (color: {intersecting_ball[4]}) will be hit at {intersection_point}")

                            # Draw impact point
                            cv2.circle(display_frame, intersection_point, 5, (0, 0, 255), -1)  # Red dot for impact

                            # Compute next trajectories
                            new_white_dir, target_ball_dir = compute_next_trajectory(intersecting_ball, intersection_point, dir_unit)

                            # --- Target Ball Trajectory (Magenta) ---
                            current_point = np.array(intersection_point, dtype=float)
                            total_length = 0

                            while total_length < max_length:
                                next_point = current_point + step * target_ball_dir

                                # Check if the target ball hits the board or falls into a hole
                                next_point, new_target_dir = check_board_collision(next_point, target_ball_dir, board_edges_mask, holes_mask)

                                if new_target_dir is None:
                                    break  # Ball falls into hole

                                current_point = next_point
                                total_length += step

                            # Draw the target ball's trajectory (Magenta)
                            cv2.line(display_frame,
                                    (int(intersection_point[0]), int(intersection_point[1])),
                                    (int(current_point[0]), int(current_point[1])),
                                    color=(255, 0, 255), thickness=4)

                            # --- New White Ball Trajectory (Green) ---
                            current_point = np.array(intersection_point, dtype=float)
                            total_length = 0

                            while total_length < max_length:
                                next_point = current_point + step * new_white_dir

                                # Check if the white ball hits the board or falls into a hole
                                next_point, new_white_dir = check_board_collision(next_point, new_white_dir, board_edges_mask, holes_mask)

                                if new_white_dir is None:
                                    break  # White ball falls into hole

                                current_point = next_point
                                total_length += step

                            # Draw the white ball's new trajectory (Green)
                            cv2.line(display_frame,
                                    (int(intersection_point[0]), int(intersection_point[1])),
                                    (int(current_point[0]), int(current_point[1])),
                                    color=(0, 255, 0), thickness=4)




            # --- End Stick Detection Code ---
            
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

            """
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
            """
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