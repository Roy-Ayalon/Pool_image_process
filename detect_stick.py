import cv2
import numpy as np

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
        return binary_k

    def detect_and_draw_stick(image, binary_k_channel):
        """
        Detect the stick line using the Fast Line Detector (LSD) and draw it on the image.
        """
        result_img = image.copy()
        # Create a line segment detector. The flag 0 means default refinement.
        lsd = cv2.createLineSegmentDetector(0)
        # Detect line segments in the binary image
        detected = lsd.detect(binary_k_channel)
        if detected[0] is None or len(detected[0]) == 0:
            print("No lines detected.")
            return result_img, None, None

        lines = detected[0]
        # Pick the longest detected line
        longest_line = None
        max_length = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > max_length:
                max_length = length
                longest_line = (int(x1), int(y1), int(x2), int(y2))

        if longest_line is None:
            print("No valid line found.")
            return result_img, None, None

        # For stability, you could add further endpoint refinement if needed.
        start_point = (longest_line[0], longest_line[1])
        end_point   = (longest_line[2], longest_line[3])
        cv2.line(result_img, start_point, end_point, (0, 0, 255), thickness=3)
        return result_img, start_point, end_point

    # Convert image to K channel and binarize it
    k_channel = convert_to_cmyk_k_channel(image, mask)
    binary_k_channel = binarize_k_channel(k_channel, threshold=220)
    # Detect and draw the stick using LSD
    result_img, start_point, end_point = detect_and_draw_stick(image, binary_k_channel)
    return result_img, start_point, end_point