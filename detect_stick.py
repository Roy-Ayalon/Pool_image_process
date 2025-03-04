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

    def detect_and_draw_stick(image, binary_k_channel):
        """
        Detect the stick line using the Fast Line Detector (LSD) and draw it on the image.
        """
        result_img = image.copy()
        lsd = cv2.createLineSegmentDetector(0)
        detected = lsd.detect(binary_k_channel)
        if detected[0] is None or len(detected[0]) == 0:
            print("No lines detected.")
            return result_img, None, None

        lines = detected[0]

        # Define a function to compute the length of a line segment.
        def line_length(line):
            # Flatten to get [x1, y1, x2, y2]
            coords = line.flatten()
            return np.linalg.norm(coords[:2] - coords[2:], 2)

        # Sort lines by their length.
        lines = sorted(lines, key=line_length)
        longest_lines = lines[-2:]  # Choose the two longest lines.

        if len(longest_lines) < 2:
            print("Not enough lines detected.")
            return result_img, None, None

        # Flatten the lines to get their endpoints.
        line1 = longest_lines[0].flatten()
        line2 = longest_lines[1].flatten()

        # Calculate the midpoints of the starting and ending endpoints.
        start_point = (int((line1[0] + line2[0]) / 2), int((line1[1] + line2[1]) / 2))
        end_point   = (int((line1[2] + line2[2]) / 2), int((line1[3] + line2[3]) / 2))

        cv2.line(result_img, start_point, end_point, (0, 0, 255), thickness=3)
        return result_img, start_point, end_point

    # Convert image to K channel and binarize it.
    k_channel = convert_to_cmyk_k_channel(image, mask)
    binary_k_channel = binarize_k_channel(k_channel, threshold=190)

    # Detect and draw the stick using LSD.
    result_img, start_point, end_point = detect_and_draw_stick(image, binary_k_channel)
    return result_img, start_point, end_point