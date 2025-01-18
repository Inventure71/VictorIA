import cv2
import numpy as np


def crop_view(image, top_left, top_right, bottom_right, bottom_left):
    # Coordinates of the corners of the distorted rectangle
    # Replace these with your actual detected points
    pts_src = np.array([
        [top_left[0], top_left[1]],  # Top-left corner
        [top_right[0], top_right[1]],  # Top-right corner
        [bottom_right[0], bottom_right[1]],  # Bottom-right corner
        [bottom_left[0], bottom_left[1]]   # Bottom-left corner
    ], dtype="float32")

    # Define the target rectangle dimensions (e.g., 700x600 pixels)
    W, H = 700, 600
    pts_dst = np.array([
        [0, 0],      # Top-left corner
        [W, 0],      # Top-right corner
        [W, H],      # Bottom-right corner
        [0, H]       # Bottom-left corner
    ], dtype="float32")

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # Apply the perspective transformation
    warped_image = cv2.warpPerspective(image, matrix, (W, H))

    # Save or display the result
    return warped_image
