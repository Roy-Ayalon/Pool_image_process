import cv2
import numpy as np
from detect_balls import detect_pool_balls
from detect_holes import detecet_holes

# Load the image
image = cv2.imread('first_pics/1.jpeg')  # Replace with your image path
if image is None:
    print('Could not open or find the image')
    exit(0)

# Convert the image to LAB color space and normalize the A channel
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab_image)
A_normalized = cv2.normalize(A, None, 0, 255, cv2.NORM_MINMAX)

# Threshold based on the LAB A channel
threshold = 80
hist, bins = np.histogram(A_normalized.flatten(), bins=256, range=(0, 255))
max_x = np.argmax(hist[:threshold])
threshold_x = 17
binary_mask = np.where((A_normalized >= (max_x - threshold_x)) & (A_normalized <= (max_x + threshold_x)), 255, 0).astype(np.uint8)

# Find contours and extract the largest one (assumed to be the table)
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)

# show polygon
visualized_image = image.copy()
cv2.drawContours(visualized_image, [largest_contour], -1, (0, 255, 0), 2)

# Show the rotated rectangle
cv2.imshow("Original Image with Rotated Rectangle", visualized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

annot, ball_info, ball_mask = detect_pool_balls('first_pics/1.jpeg')

# binary mask of the rotated rectangle
mask = np.zeros_like(binary_mask)
cv2.drawContours(mask, [largest_contour], -1, 255, -1)

# Show the binary mask of the table
cv2.imshow("Binary Mask of the Table", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# remove the balls from the mask
mask_without_balls = cv2.bitwise_and(mask, 255 - ball_mask)

# Show the mask of the table without balls
cv2.imshow("Mask of the table without balls", mask_without_balls)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detect holes
holes_mask = detecet_holes(image, mask_without_balls)

# add mask of the holes to the mask of the table
final_mask = cv2.bitwise_or(mask, holes_mask)

# Show the final mask
cv2.imshow("Final Mask", final_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


