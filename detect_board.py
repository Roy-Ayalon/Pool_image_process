import cv2
import numpy as np
from detect_balls import detect_pool_balls
from detect_holes import detecet_holes
import cv2
import numpy as np
from detect_balls import detect_pool_balls
from detect_holes import detecet_holes
import matplotlib.pyplot as plt

def detect_board(frame, debug=False):
    # Convert the image to LAB color space and normalize the A channel
    lab_image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab_image)
    A_normalized = cv2.normalize(A, None, 0, 255, cv2.NORM_MINMAX)

    if debug:
        cv2.imshow("Normalized A Channel", A_normalized)
        cv2.waitKey(0)

    # Threshold based on the LAB A channel
    #range_min = 30
    #range_max = 80
    threshold = 45
    hist, bins = np.histogram(A_normalized.flatten(), bins=256, range=(0, 255))
    max_x = np.argmax(hist[:threshold])
    threshold_x = 25
    binary_mask = np.where((A_normalized >= (max_x - threshold_x)) & 
                           (A_normalized <= (max_x + threshold_x)), 
                           255, 0).astype(np.uint8)
    
    #binary_mask = cv2.inRange(A_normalized, range_min, range_max)

    # show histogram
    if debug:
        plt.plot(hist)
        plt.show()

    if debug:
        cv2.imshow("Binary Mask", binary_mask)
        cv2.waitKey(0)

    # Find contours and extract the largest one (assumed to be the table)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        if debug:
            print("No contours found.")
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    if debug:
        visualized_image = frame.copy()
    black_image = np.zeros_like(frame) #### update
    cv2.drawContours(black_image, [largest_contour], -1, (0, 255, 0), 2) ## update
        
        # Optionally display all contours
        #all_contours_img = frame.copy()
        #cv2.drawContours(all_contours_img, contours, -1, (255, 0, 0), 2)
        #cv2.imshow("All Contours", all_contours_img)
        #cv2.waitKey(0)

    return largest_contour, black_image

def find_hole_centers(holes_contours):
    """
    Given a list of hole contours, compute the center (centroid) of each hole.
    Returns a list of (x, y) coordinates.
    """
    centers = []
    for cnt in holes_contours:
        # Use moments to compute centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
        else:
            # Fallback: average of points if area is zero
            coords = cnt[:, 0, :]
            cx = int(np.mean(coords[:, 0]))
            cy = int(np.mean(coords[:, 1]))
            centers.append((cx, cy))
    return centers

def find_boundary_points_for_hole(board_contour, hole_center):
    """
    For a given hole center (x, y) and a board contour,
    find the closest intersection points on the left and right boundaries.
    Returns (left_point, right_point) where each is a tuple (x, y).
    """
    hx, hy = hole_center
    left_point = None
    right_point = None
    min_left_dist = float('inf')
    min_right_dist = float('inf')

    # Assume board_contour is an array of points forming a closed polygon
    points = board_contour.reshape(-1, 2)
    num_points = len(points)

    for i in range(num_points):
        # Current segment points
        x1, y1 = points[i]
        x2, y2 = points[(i+1) % num_points]

        # Check if horizontal line through hole center crosses this edge
        if (y1 - hy) * (y2 - hy) <= 0:  # segment crosses horizontal line at hy
            # Avoid division by zero for horizontal segments
            if y2 != y1:
                # Linear interpolation to find x where y = hy on the segment
                t = (hy - y1) / (y2 - y1)
                inter_x = x1 + t * (x2 - x1)
                inter_point = (int(inter_x), hy)

                # Determine if this point is left or right of hole center
                if inter_x < hx:
                    dist = hx - inter_x
                    if dist < min_left_dist:
                        min_left_dist = dist
                        left_point = inter_point
                else:
                    dist = inter_x - hx
                    if dist < min_right_dist:
                        min_right_dist = dist
                        right_point = inter_point

    return left_point, right_point

def find_all_boundary_points(board_contour, hole_centers):
    """
    For each hole center, find corresponding left and right boundary points.
    Returns a dict mapping hole center to its (left_point, right_point).
    """
    boundary_points = {}
    for center in hole_centers:
        left_pt, right_pt = find_boundary_points_for_hole(board_contour, center)
        boundary_points[center] = {'left': left_pt, 'right': right_pt}
    return boundary_points

def find_intermediate_boundary_points(board_contour, hole_centers, boundary_points):
    """
    Given hole centers sorted along an axis and their corresponding boundary points,
    find intermediate points on the board boundary between each pair of neighbor holes.
    Returns a list of tuples with structure:
       ((hole1_center, hole2_center), (intermediate_left, intermediate_right))
    where intermediate_left/right are points on the boundary between holes.
    """
    # Sort holes, for example by y-coordinate if arranged vertically.
    sorted_holes = sorted(hole_centers, key=lambda pt: pt[1])
    intermediates = []

    for i in range(len(sorted_holes)-1):
        hole1 = sorted_holes[i]
        hole2 = sorted_holes[i+1]

        # Retrieve known boundary points for the two holes
        bp1 = boundary_points.get(hole1)
        bp2 = boundary_points.get(hole2)
        if not bp1 or not bp2:
            continue

        # Example: Find midpoint between right boundary of hole1 and left boundary of hole2
        # as an intermediate point on the boundary between these holes.
        # You can refine how you choose these intermediate points.
        if bp1['right'] and bp2['left']:
            inter_x = (bp1['right'][0] + bp2['left'][0]) // 2
            inter_y = (bp1['right'][1] + bp2['left'][1]) // 2
            intermediate_point = (inter_x, inter_y)
        else:
            intermediate_point = None

        intermediates.append(((hole1, hole2), intermediate_point))

    return intermediates




# frame = cv2.imread('/Users/mikitatarjitzky/Documents/DIP_project/WhatsApp Image 2024-12-29 at 09.40.29 (2).jpeg')
# board_with_mask = detect_board(frame)

# cv2.imshow("Board with Mask", board_with_mask)
# # Show the binary mask of the table
# cv2.imshow("Binary Mask of the Table", mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # remove the balls from the mask
# mask_without_balls = cv2.bitwise_and(mask, 255 - ball_mask)

# # Show the mask of the table without balls
# cv2.imshow("Mask of the table without balls", mask_without_balls)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Detect holes
# holes_mask = detecet_holes(image, mask_without_balls)

# # add mask of the holes to the mask of the table
# final_mask = cv2.bitwise_or(mask, holes_mask)

# # Show the final mask
# cv2.imshow("Final Mask", final_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


