import cv2
import numpy as np
from matplotlib import pyplot as plt

def table_start(image, threshold):
    # Normalize the image to the range [0, 1]
    image_rgb = image.astype(float) / 255.0

    # Compute the K channel
    K = 1 - np.max(image_rgb, axis=2)
    K = (K * 255).astype(np.uint8)  # Scale K back to 0-255 range

    # Threshold K to create a binary mask
    K_binary = (K >= threshold).astype(np.uint8)  # Mask: 1 where K < threshold, 0 otherwise

    return K_binary
