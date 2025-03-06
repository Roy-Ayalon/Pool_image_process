import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_stick(image, mask):
    def convert_to_cmyk_k_channel(image, mask):
        """
        Convert the input image to the CMYK color model and return the K (black) channel.
        Areas where the mask is zero are replaced with a replacement color.
        """
        replacement_color = np.array([100, 100, 100], dtype=np.uint8)
        # Use np.where to replace background pixels based on the mask
        image_proc = np.where(mask[..., None] == 0, replacement_color, image)
        image_rgb = image_proc.astype(float) / 255.0
        # K channel: 1 - max(R, G, B)
        K = 1 - np.max(image_rgb, axis=2)
        K = (K * 255).astype(np.uint8)
        return K

    def binarize_k_channel(k_channel, threshold=127):
        """
        Binarize the K channel using a threshold.
        """
        _, binary_k = cv2.threshold(k_channel, threshold, 255, cv2.THRESH_BINARY)
        # show the binary image
        #plt.imshow(binary_k, cmap='gray')
        #plt.show()

        return binary_k

    def detect_and_draw_stick(image, binary_k_channel, min_line_length=30, perp_threshold=150, inlier_threshold=150):
        """
        Detect and draw a stable stick on the image with debug visualization at various stages.
        """
        import numpy as np
        import cv2

        result_img = image.copy()
        lsd = cv2.createLineSegmentDetector(0)
        detected = lsd.detect(binary_k_channel)

        if detected[0] is None or len(detected[0]) == 0:
            print("No lines detected.")
            return result_img, None, None

        lines = detected[0]

        # Helper: compute length of a line segment.
        def line_length(line):
            coords = line.flatten()  # [x1, y1, x2, y2]
            return np.linalg.norm(coords[:2] - coords[2:], 2)

        # --- Debug: Draw all detected lines ---
        debug_img = result_img.copy()
        for line in lines:
            x1, y1, x2, y2 = line.flatten()
            cv2.line(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=1)
        #cv2.imshow("Detected Lines", debug_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Filter out short lines.
        valid_lines = [line for line in lines if line_length(line) >= min_line_length]
        if not valid_lines:
            print("No valid lines detected (all lines are too short).")
            return result_img, None, None

        # If only one valid line exists, return it.
        if len(valid_lines) == 1:
            line = valid_lines[0].flatten()
            start_point = (int(line[0]), int(line[1]))
            end_point   = (int(line[2]), int(line[3]))
            #cv2.line(result_img, start_point, end_point, (0, 0, 255), thickness=3)
            #cv2.imshow("Single Valid Line", result_img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            return result_img, start_point, end_point

        # --- Debug: Draw all valid lines ---
        debug_img = result_img.copy()
        for line in valid_lines:
            x1, y1, x2, y2 = line.flatten()
            cv2.line(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=1)
        #cv2.imshow("Valid Lines", debug_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Compute the angle for each valid line.
        angles = []
        for line in valid_lines:
            x1, y1, x2, y2 = line.flatten()
            angle = np.arctan2(y2 - y1, x2 - x1)
            angles.append(angle)
        median_angle = np.median(angles)

        # Define unit vectors: u along median angle and v its perpendicular.
        u = np.array([np.cos(median_angle), np.sin(median_angle)])
        v = np.array([-np.sin(median_angle), np.cos(median_angle)])

        # Collect all endpoints and project onto u and v.
        points = []
        p_vals = []  # along u
        q_vals = []  # along v
        for line in valid_lines:
            pts = line.flatten()
            for i in range(0, 4, 2):
                pt = np.array([pts[i], pts[i+1]])
                points.append(pt)
                p_vals.append(pt.dot(u))
                q_vals.append(pt.dot(v))
        points = np.array(points)
        p_vals = np.array(p_vals)
        q_vals = np.array(q_vals)

        # Filter out endpoints that deviate too much in the perpendicular direction.
        q_med = np.median(q_vals)
        perp_mask = np.abs(q_vals - q_med) <= perp_threshold
        if np.sum(perp_mask) < 2:
            print("Not enough points in the central stick cluster (perp filter).")
            return result_img, None, None
        points_filtered = points[perp_mask]

        # --- Debug: Visualize endpoints after perpendicular filtering ---
        debug_img = result_img.copy()
        for pt in points:
            # Green if within threshold, red if not.
            color = (0, 255, 0) if abs(pt.dot(v) - q_med) <= perp_threshold else (0, 0, 255)
            cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 3, color, -1)
        #cv2.imshow("Perp Filter Points", debug_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Robustly fit a line using the filtered points.
        if len(points_filtered) >= 2:
            pts_for_fit = points_filtered.reshape(-1, 1, 2).astype(np.float32)
            [vx, vy, x0, y0] = cv2.fitLine(pts_for_fit, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        else:
            print("Not enough points for robust line fitting.")
            return result_img, None, None

        # Further filter points based on distance to the fitted line.
        # Distance from a point (x,y) to line through (x0,y0) with direction (vx,vy):
        dists = np.abs(vy * (points_filtered[:,0] - x0) - vx * (points_filtered[:,1] - y0))
        inlier_mask = dists <= inlier_threshold
        final_points = points_filtered[inlier_mask]
        if len(final_points) < 2:
            print("Not enough inlier points after line fitting.")
            return result_img, None, None

        # --- Debug: Visualize inlier points and the fitted line ---
        debug_img = result_img.copy()
        for pt in final_points:
            cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 4, (255, 0, 0), -1)
        # Extend the fitted line across the image.
        height, width = result_img.shape[:2]
        #Determine two far points along the line.
        extension_length = 50  # extend 50 pixels on each side
        pt_left = (int(x0 - extension_length * vx), int(y0 - extension_length * vy))
        pt_right = (int(x0 + extension_length * vx), int(y0 + extension_length * vy))
        cv2.line(debug_img, pt_left, pt_right, (255, 255, 255), 2)

        #cv2.imshow("Inlier Points and Fitted Line", debug_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Draw a circle around the point (x0, y0) in white
        #cv2.circle(debug_img, (int(x0), int(y0)), 5, (255, 255, 255), -1)
        #cv2.imshow("Circle around (x0, y0)", debug_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()



       

        
        start_point = pt_left
        end_point = pt_right

        # Verify the final stick is long enough.
        if np.linalg.norm(np.array(start_point) - np.array(end_point)) < min_line_length:
            print("Resulting stick is too short.")
            return result_img, None, None

        cv2.line(result_img, start_point, end_point, (0, 0, 255), thickness=3)
        
        # --- Debug: Show final result ---
        #cv2.imshow("Final Stick", result_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()



        return result_img, start_point, end_point

    # Convert image to K channel and binarize it.
    k_channel = convert_to_cmyk_k_channel(image, mask)
    binary_k_channel = binarize_k_channel(k_channel, threshold=210)

    # Detect and draw the stick using LSD.
    result_img, start_point, end_point = detect_and_draw_stick(image, binary_k_channel)
    return result_img, start_point, end_point