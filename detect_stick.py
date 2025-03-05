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

    def detect_and_draw_stick(image, binary_k_channel, min_line_length=50, perp_threshold=150, inlier_threshold=150):
        """
        Detect and draw a stable stick on the image.

        This function uses the Fast Line Detector (LSD) to detect line segments in the 
        binary_k_channel. It filters out segments shorter than a specified threshold,
        computes the median angle of the remaining segments, and projects all endpoints
        onto the median direction and its perpendicular. Endpoints too far (perp_threshold)
        from the central cluster are discarded. Then, a robust line is fitted (via cv2.fitLine)
        to the remaining points and further refined by rejecting points with high distance 
        from the fitted line (inlier_threshold). Finally, the extreme endpoints along the
        fitted line direction define the stick.

        Parameters:
            image (numpy.ndarray): The original image.
            binary_k_channel (numpy.ndarray): A binary image (e.g., one channel) for LSD.
            min_line_length (float): Minimum length for a line segment to be considered.
            perp_threshold (float): Maximum allowed deviation in the perpendicular direction.
            inlier_threshold (float): Maximum allowed distance from the robustly fitted line.

        Returns:
            result_img (numpy.ndarray): The image with the detected stick drawn.
            start_point (tuple or None): (x, y) coordinates for one end of the stick, or None if not found.
            end_point (tuple or None): (x, y) coordinates for the other end of the stick, or None if not found.
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
            cv2.line(result_img, start_point, end_point, (0, 0, 255), thickness=3)
            return result_img, start_point, end_point

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

        # Robustly fit a line using the filtered points.
        if len(points_filtered) >= 2:
            pts_for_fit = points_filtered.reshape(-1, 1, 2).astype(np.float32)
            [vx, vy, x0, y0] = cv2.fitLine(pts_for_fit, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        else:
            print("Not enough points for robust line fitting.")
            return result_img, None, None

        # Further filter points based on distance to the fitted line.
        # The distance from a point (x, y) to the line through (x0,y0) with direction (vx,vy):
        # distance = |vy*(x - x0) - vx*(y - y0)|
        dists = np.abs(vy * (points_filtered[:,0] - x0) - vx * (points_filtered[:,1] - y0))
        inlier_mask = dists <= inlier_threshold
        final_points = points_filtered[inlier_mask]
        if len(final_points) < 2:
            print("Not enough inlier points after line fitting.")
            return result_img, None, None

        # Use the fitted line direction for projection.
        line_dir = np.array([vx, vy])
        # Project final points onto the line.
        projections = final_points.dot(line_dir)
        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)
        start_point = tuple(final_points[min_idx].astype(int))
        end_point = tuple(final_points[max_idx].astype(int))

        # Verify the final stick is long enough.
        if np.linalg.norm(np.array(start_point) - np.array(end_point)) < min_line_length:
            print("Resulting stick is too short.")
            return result_img, None, None

        cv2.line(result_img, start_point, end_point, (0, 0, 255), thickness=3)
        return result_img, start_point, end_point

    # Convert image to K channel and binarize it.
    k_channel = convert_to_cmyk_k_channel(image, mask)
    binary_k_channel = binarize_k_channel(k_channel, threshold=210)

    # Detect and draw the stick using LSD.
    result_img, start_point, end_point = detect_and_draw_stick(image, binary_k_channel)
    return result_img, start_point, end_point