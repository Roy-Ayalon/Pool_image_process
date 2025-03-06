import cv2
import numpy as np

from detect_balls import detect_pool_balls, detect_white_ball
from detect_board import detect_board
from detect_holes import detecet_holes
from detect_stick import detect_stick
from trajectory import okay_to_shoot, compute_trajectory, extend_line, find_first_intersecting_ball, compute_next_trajectory, check_hole_intersection, check_board_edge_intersection
import matplotlib.pyplot as plt
from ball_panel import create_balls_panel
from is_moving import check_white_ball_movement  # Updated function signature

# Global persistent dictionary for remaining balls.
remaining_balls = {}

# Global point counter.
points = 0

# Global variable to store the previous white ball center.
previous_white_ball_center = None

balls_info = None

black_missimg_counter = 0
score_counter = 0
win_counter = 0

displayed_score = None
candidate_score = None
stable_frame_count = 0

# New global variable for turn mode cooldown:
turn_mode_cooldown = 0  # frames to wait before allowing next turn

def main():
    global remaining_balls, points, previous_white_ball_center, balls_info, black_missimg_counter, score_counter, win_counter, displayed_score, candidate_score, stable_frame_count, turn_mode_cooldown # Declare globals

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
        frame_blurred = cv2.GaussianBlur(frame, (5,5), 0)
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
            white_ball = detect_white_ball(frame_blurred, board_contour)
            
            if board_contour is not None:
                cv2.drawContours(display_frame, [board_contour], -1, (0, 0, 255), 2)
            
            if balls_contour is not None and len(balls_contour) > 0:
                for ball_info in balls_info:
                    x, y, r, ball_label, _ = ball_info
                    cv2.circle(display_frame, (x, y), r, (0, 0, 255), 2)
                    cv2.putText(display_frame, ball_label, (x - r, y - r - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if white_ball is not None:
                wx, wy, wr = white_ball
                cv2.circle(display_frame, (wx, wy), wr, (0, 0, 255), 2)
                cv2.putText(display_frame, 'white', (wx - wr, wy - wr - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(display_frame, "Balls Detected: Press 's' to start game", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Live Video", display_frame)

        elif mode == "game":
            # Re-detect balls (or track them) in game mode.
            annotated, balls_info, ball_mask, balls_contour, binary_balls = detect_pool_balls(frame_blurred, board_contour)
            white_ball = detect_white_ball(frame_blurred, board_contour)

            if board_contour is not None:
                cv2.drawContours(display_frame, [board_contour], -1, (0, 0, 255), 2)
            
            if balls_contour is not None and len(balls_contour) > 0:
                for ball_info in balls_info:
                    x, y, r, ball_label, _ = ball_info
                    cv2.circle(display_frame, (x, y), r, (0, 0, 255), 2)
                    cv2.putText(display_frame, ball_label, (x - r, y - r - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
            if white_ball is not None:
                wx, wy, wr = white_ball
                cv2.circle(display_frame, (wx, wy), wr, (0, 0, 255), 2)
                cv2.putText(display_frame, 'white', (wx - wr, wy - wr - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
            # Decrement cooldown if active.
            if turn_mode_cooldown > 0:
                turn_mode_cooldown -= 1

            # --- White Ball Movement Check ---
            # First, detect the white ball.
            white_ball_moving = False
            if white_ball is not None:
                white_center = (white_ball[0], white_ball[1])
                white_ball_moving = check_white_ball_movement(white_center,previous_white_ball_center, frame, threshold=20)
                previous_white_ball_center = white_center
            
            if white_ball_moving:
                # FOR DEBUGGING
                cv2.putText(display_frame, "White ball moving - stick detection skipped", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # --- Stick Detection Code ---
                res_img, start_point, end_point = detect_stick(frame, binary_mask)
                if start_point is not None and end_point is not None:
                    cv2.line(display_frame, start_point, end_point, (255, 0, 0), 3)

                    # --- physics ---
                    line = (start_point, end_point)

                    # do pyhsics only if there is a white ball and there is a stick pointed to it
                    if white_ball is not None:
                        okay = okay_to_shoot(display_frame, line, white_ball, balls_info)
                        # # Display the text in the center of the frame
                        # # FOR DEBUGGING
                        # overlay_text = "True" if okay else "False"
                        # text_color = (0, 255, 0) if okay else (0, 0, 255)  # Green for True, Red for False
                        # font = cv2.FONT_HERSHEY_SIMPLEX
                        # font_scale = 1.5
                        # thickness = 3
                        # text_size = cv2.getTextSize(overlay_text, font, font_scale, thickness)[0]
                        # text_x = (frame.shape[1] - text_size[0]) // 2
                        # text_y = (frame.shape[0] + text_size[1]) // 2
                        # cv2.putText(display_frame, overlay_text, (text_x, text_y), font, font_scale, text_color, thickness)

                        if okay and turn_mode_cooldown == 0:
                            mode = 'turn'
                            continue
   
            cv2.putText(display_frame, "Game Mode: Press 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # --- scoring section ---
            colored_balls = [ball for ball in balls_info if ball[3].lower() not in ("white", "black")]
            new_score = 10 - len(colored_balls)

            # On the very first frame, initialize both displayed_score and candidate_score.
            if displayed_score is None:
                displayed_score = new_score
                candidate_score = new_score
                stable_frame_count = 0
            else:
                if new_score != displayed_score:
                    # A new score candidate is detected.
                    if candidate_score == new_score:
                        stable_frame_count += 1
                    else:
                        candidate_score = new_score
                        stable_frame_count = 1

                    if stable_frame_count > 4:
                        displayed_score = candidate_score
                else:
                    # New score matches the currently displayed score.
                    candidate_score = new_score
                    stable_frame_count = 0

            # Always display the current displayed_score.
            text = f"Score: {displayed_score}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_x = (display_frame.shape[1] - text_width) // 2
            cv2.putText(display_frame, text, (text_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # # --- Check win/loss conditions ---
            if new_score == 10:
                win_counter +=1
            else:
                win_counter = 0

            if win_counter > 7:
                h, w = display_frame.shape[:2]
                cv2.putText(display_frame, "YOU WON", (w//2 - 150, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
                cv2.imshow("Live Video", display_frame)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('w'):
                    # Reset game state.
                    new_score = 0
                    mode = "game"
                    continue
                if key == ord('q'):
                    break

            # Else, if the black ball is missing (pocketed) but not all colored balls are pocketed, you lose.
            black_found = any(ball[3].lower() == "black" for ball in balls_info)
            if not black_found:
                black_missing_counter += 1
            else:
                # Reset the counter if black ball is detected.
                black_missing_counter = 0

            # Once we've missed the black ball for more than threshold frames, trigger game over.
            if black_missing_counter > 7:
                h, w = display_frame.shape[:2]
                cv2.putText(display_frame, "YOU LOSE", (w//2 - 150, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 4)
                cv2.imshow("Live Video", display_frame)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('l'):
                    new_score = 0
                    mode = "game"
                    continue
                if key == ord('q'):
                    break

            present_balls = [ball for ball in balls_info if ball[3].lower() not in ("white")]
            if present_balls:
                h, w = display_frame.shape[:2]
                panel = create_balls_panel(present_balls, w, panel_height=100)
                combined_frame = np.vstack([display_frame, panel])
                cv2.imshow("Live Video", combined_frame)
            else:
                cv2.imshow("Live Video", display_frame)

        elif mode == 'turn':
            text = 'Turn Mode - HIT THE WHITE BALL!'
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_x = (display_frame.shape[1] - text_width) // 2
            cv2.putText(display_frame, text, (text_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if board_contour is not None:
                cv2.drawContours(display_frame, [board_contour], -1, (0, 0, 255), 2)

            white_ball = detect_white_ball(frame_blurred, board_contour)

            if white_ball is not None:
                wx, wy, wr = white_ball
                cv2.circle(display_frame, (wx, wy), wr, (0, 0, 255), 2)
                cv2.putText(display_frame, 'white', (wx - wr, wy - wr - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                white_center = (wx, wy)
                white_ball_moving = check_white_ball_movement(white_center,previous_white_ball_center, frame, threshold=5)
                previous_white_ball_center = white_center

            for ball_info in balls_info:
                    x, y, r, ball_label, _ = ball_info
                    cv2.circle(display_frame, (x, y), r, (0, 0, 255), 2)
                    cv2.putText(display_frame, ball_label, (x - r, y - r - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
            if white_ball_moving:
                turn_mode_cooldown = 20
                mode = 'game'
                continue

            else:        
                res_img, start_point, end_point = detect_stick(frame, binary_mask)
                if start_point is not None and end_point is not None:
                    cv2.line(display_frame, start_point, end_point, (255, 0, 0), 3)

                    line = (start_point, end_point)
                    if white_ball is not None:
                        okay = okay_to_shoot(display_frame, line, white_ball, balls_info)
                        if okay:
                            white_x, white_y, white_radius = white_ball
                            white_center = (white_x, white_y)

                            # Compute initial trajectory

                            # Compute initial trajectory
                            dir_unit, contact_point = compute_trajectory(white_center, white_radius, line)

                            # --- White Ball First Trajectory ---
                            intersecting_ball, intersection_point = find_first_intersecting_ball(contact_point, dir_unit, balls_info, board_contour)

                            # --- check if the white ball hit a hole ---
                            hole_hit, _, _ = check_hole_intersection(contact_point, dir_unit, holes_mask)

                            # --- check if the white ball hit the board edges ---
                            board_hit, _, _, board_hit_point, reflection_dir = check_board_edge_intersection(contact_point, dir_unit, board_perimeter_mask, max_length=100000, line_thickness=5)

                            step = 5  # Small step size for trajectory advancement
                            current_point = np.array(contact_point, dtype=float)
                            total_length = 0
                            max_length = 1000

                            # --- case 1 - hitting colored ball ---
                            if intersecting_ball:
                                trajectory_end = np.array(intersection_point, dtype=float)  # Stop at the ball

                                # Draw the white ball's trajectory
                                cv2.line(display_frame,
                                        (int(contact_point[0]), int(contact_point[1])),
                                        (int(trajectory_end[0]), int(trajectory_end[1])),
                                        color=(255, 255, 0), thickness=4)
                                
                                # Draw the colored ball's trajectory
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

                                    # Stop if outside board contour
                                    if cv2.pointPolygonTest(board_contour, (next_point[0], next_point[1]), False) < 0:
                                        break

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

                                    # Stop if outside board contour
                                    if cv2.pointPolygonTest(board_contour, (next_point[0], next_point[1]), False) < 0:
                                        break

                                    current_point = next_point
                                    total_length += step

                                # Draw the white ball's new trajectory (Green)
                                cv2.line(display_frame,
                                        (int(intersection_point[0]), int(intersection_point[1])),
                                        (int(current_point[0]), int(current_point[1])),
                                        color=(255, 255, 0), thickness=4)

                            # --- case 2 - hitting a hole ---       
                            elif hole_hit:
                                total_length = 0
                                max_length = 1000
                                current_point = np.array(contact_point, dtype=float)
                                while total_length < max_length:
                                    next_point = current_point + step * dir_unit

                                    # Check if the next point is still inside the board contour
                                    if cv2.pointPolygonTest(board_contour, (next_point[0], next_point[1]), False) < 0:
                                        break  # Stop drawing when exiting the board

                                    current_point = next_point
                                    total_length += step

                                trajectory_end = current_point  # Stop at the board

                                # Draw the white ball's trajectory (Yellow)
                                cv2.line(display_frame,
                                        (int(contact_point[0]), int(contact_point[1])),
                                        (int(trajectory_end[0]), int(trajectory_end[1])),
                                        color=(255, 255, 0), thickness=4)
                            
                            # --- case 3 - hitting the board edges ---
                            elif board_hit:
                                #cv2.circle(display_frame, board_hit_point, 5, (0, 0, 255), -1)  # Red dot for impact

                                current_point = np.array(contact_point, dtype=float)
                                while total_length < max_length:
                                    next_point = current_point + step * dir_unit

                                    # Check if the next point is still inside the board contour
                                    if cv2.pointPolygonTest(board_contour, (next_point[0], next_point[1]), False) < 0:
                                        break  # Stop drawing when exiting the board

                                    current_point = next_point
                                    total_length += step

                                trajectory_end = current_point  # Stop at the board
                                # Draw the white ball's trajectory (Yellow)
                                cv2.line(display_frame,
                                        (int(contact_point[0]), int(contact_point[1])),
                                        (int(trajectory_end[0]), int(trajectory_end[1])),
                                        color=(255, 255, 0), thickness=4)

                                total_length = 0
                                max_length = 1000
                                current_point = np.array(board_hit_point, dtype=float)
                                while total_length < max_length:
                                    next_point = current_point + step * reflection_dir

                                    # Check if the next point is still inside the board contour
                                    if cv2.pointPolygonTest(board_contour, (next_point[0], next_point[1]), False) < 0:
                                        break  # Stop drawing when exiting the board

                                    current_point = next_point
                                    total_length += step

                                trajectory_end = current_point  # Stop at the board

                                # Draw the white ball's trajectory (Yellow)
                                cv2.line(display_frame,
                                        (int(board_hit_point[0]), int(board_hit_point[1])),
                                        (int(trajectory_end[0]), int(trajectory_end[1])),
                                        color=(255, 255, 0), thickness=4)

                            else:
                                # If no ball is hit, continue to the board contour
                                while total_length < max_length:
                                    next_point = current_point + step * dir_unit

                                    # Check if the next point is still inside the board contour
                                    if cv2.pointPolygonTest(board_contour, (next_point[0], next_point[1]), False) < 0:
                                        break  # Stop drawing when exiting the board

                                    current_point = next_point
                                    total_length += step

                                # Draw the white ball's trajectory (Yellow)
                                cv2.line(display_frame,
                                        (int(contact_point[0]), int(contact_point[1])),
                                        (int(trajectory_end[0]), int(trajectory_end[1])),
                                        color=(255, 255, 0), thickness=4)
                    
            cv2.imshow("Live Video", display_frame)
                
            # for debugging
            # key = cv2.waitKey(33) & 0xFF
            # if key == ord('r'):
            #     break
            
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