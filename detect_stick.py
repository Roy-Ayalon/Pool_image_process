import cv2
import numpy as np
from matplotlib import pyplot as plt

def detect_stick(image):
    def convert_to_cmyk_k_channel(image):
        """
        Convert the input image to the CMYK color model and return the K (black) channel.
        """
        image_rgb = image.astype(float) / 255.0
        K = 1 - np.max(image_rgb, axis=2)
        K = (K * 255).astype(np.uint8)  # Scale K back to 0-255 range
        return K

    def binarize_k_channel(k_channel, threshold=127):
        """
        Binarize the K channel using a threshold.
        """
        _, binary_k = cv2.threshold(k_channel, threshold, 255, cv2.THRESH_BINARY)
        return binary_k

    def refine_stick_line(line, binary_mask, max_gap=10, max_extension=1000, extra_length=50):
        """
        Refine the detected stick line by finding the exact start and end points
        in the binary mask, extending the line beyond the initially detected segment.

        Parameters:
            line (tuple): Detected line (x1, y1, x2, y2).
            binary_mask (np.ndarray): Binary mask of the image.
            max_gap (int): Maximum allowed gap (number of consecutive non-white points).
            max_extension (int): Maximum number of points to extend the line beyond its original bounds.

        Returns:
            start_point (tuple): Refined start point of the stick.
            end_point (tuple): Refined end point of the stick.
        """
        def extend_line(x, y, dx, dy, binary_mask, direction):
            """
            Extend the line in a specific direction.
            """
            consecutive_non_white = 0
            extended_point = (x, y)

            for _ in range(max_extension):
                x, y = x + dx * direction, y + dy * direction
                if not (0 <= int(x) < binary_mask.shape[1] and 0 <= int(y) < binary_mask.shape[0]):
                    break  # Stop if out of bounds

                if binary_mask[int(y), int(x)] == 255:  # White in binary mask
                    consecutive_non_white = 0
                    extended_point = (int(x), int(y))
                else:
                    consecutive_non_white += 1

                if consecutive_non_white > max_gap:  # Stop extending if too many non-white points
                    break

            return extended_point

        x1, y1, x2, y2 = line

        # Calculate direction vector
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        dx /= length  # Normalize direction vector
        dy /= length

        # Extend in both directions
        start_point = extend_line(x1, y1, dx, dy, binary_mask, direction=-1)
        end_point = extend_line(x2, y2, dx, dy, binary_mask, direction=1)

        # Add extra constant length to both ends
        extended_start_point = (int(start_point[0] - dx * extra_length), int(start_point[1] - dy * extra_length))
        extended_end_point = (int(end_point[0] + dx * extra_length), int(end_point[1] + dy * extra_length))

        return extended_start_point, extended_end_point


    def detect_and_refine_stick(image, binary_k_channel):
        """
        Detect the stick line and refine its endpoints based on the binary mask.
        """
        # Edge detection
        edges = cv2.Canny(binary_k_channel, 150, 250)

        # Detect line segments using Probabilistic Hough Transform
        line_segments = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=100)

        if line_segments is None:
            print("No lines detected.")
            return image, None

        # Find the thickest line
        thickest_line = None
        max_length = 0

        for line in line_segments:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > max_length:
                max_length = length
                thickest_line = (x1, y1, x2, y2)

        if not thickest_line:
            print("No valid line found.")
            return image, None

        # Refine the line by finding exact start and end points in the binary mask
        start_point, end_point = refine_stick_line(thickest_line, binary_k_channel)

        # Draw the refined line on the original image
        result_img = image.copy()
        if start_point and end_point:
            cv2.line(result_img, start_point, end_point, (0, 255, 0), thickness=3)  # Green line
        else:
            print("Could not refine the stick line.")

        return result_img, (start_point, end_point)
    
    # Convert to the K channel of CMYK
    k_channel = convert_to_cmyk_k_channel(image)

    # Binarize the K channel
    binary_k_channel = binarize_k_channel(k_channel, threshold=200)

    # Detect and refine the stick line
    result_img, refined_line = detect_and_refine_stick(image, binary_k_channel)

    return refined_line

# def detect_stick(image, binary_k_channel):
#     """
#     Detect the thickest line using the Probabilistic Hough Line Transform.
#     """
#     # Apply edge detection to the binary K channel
#     edges = cv2.Canny(binary_k_channel, 150, 250)

#     # Perform Probabilistic Hough Line Transform
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=210)
#     if lines is None:
#         print("No lines detected.")
#         return image, None, edges

#     # Identify the thickest line
#     thickest_line = None
#     max_thickness = 0

#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         line_mask = np.zeros_like(edges, dtype=np.uint8)
#         cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=5)  # Draw a thick line
#         intersection = cv2.bitwise_and(line_mask, edges)  # Measure how much edge aligns with this line
#         thickness = np.sum(intersection == 255)  # Count white pixels on the mask

#         if thickness > max_thickness:
#             max_thickness = thickness
#             thickest_line = (x1, y1, x2, y2)

#     # Draw the thickest line on the original image
#     result_img = image.copy()
#     if thickest_line:
#         x1, y1, x2, y2 = thickest_line
#         cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)  # Green line

#     return result_img, thickest_line, edges

# # Load the image
# image_path = "/Users/mikitatarjitzky/Documents/DIP_project/stick3.jpg"  # Replace with your image path
# image = cv2.imread(image_path)

# # Convert to the K channel of CMYK
# k_channel = convert_to_cmyk_k_channel(image)

# # Binarize the K channel
# binary_k_channel = binarize_k_channel(k_channel, threshold=200)

# # Detect and refine the stick line
# result_img, refined_line = detect_and_refine_stick(image, binary_k_channel)

# # Display results
# plt.figure(figsize=(15, 8))

# plt.subplot(1, 3, 1)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title("Original Image")
# plt.axis("off")

# plt.subplot(1, 3, 2)
# plt.imshow(binary_k_channel, cmap="gray")
# plt.title("Processed Binary K Channel")
# plt.axis("off")

# plt.subplot(1, 3, 3)
# plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
# plt.title("Refined Stick Line")
# plt.axis("off")

# plt.tight_layout()
# plt.show()

# if refined_line:
#     print(f"Refined Stick Line: Start = {refined_line[0]}, End = {refined_line[1]}")
# else:
#     print("No stick line could be refined.")
