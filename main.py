import cv2
import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib.pyplot as plt

from ImagePointSelection import ImageClick
from utils.sam_model_handler import SamModelHandler
from utils.calibration_utils import show_mask, show_points
from utils.calculate_intersection import calculate_intersection
from utils.border_detection import BorderDetector
from utils.camera_calibration import crop_view


def handle_points_labels(points, labels, path):
    global input_point, input_label, image_path
    input_point = points
    input_label = labels
    image_path = path


def main():
    USE_WEBCAM = False
    player1_color = (88, 168, 55)
    player2_color = (213, 84, 89)

    # Initialize tkinter for image selection
    root = tk.Tk()
    app = ImageClick(root, handle_points_labels, USE_WEBCAM)
    root.mainloop()
    if not image_path:
        print("No image selected. Exiting.")
        return

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # draw these points on the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    show_points(input_point, input_label, plt.gca())
    plt.show()

    # Handle SAM model
    sam_handler = SamModelHandler("Models/SAM Model Checkpoint.pth")
    sam_handler.set_image(image)

    # Predict masks
    masks, scores, _ = sam_handler.predict(input_point, input_label)
    for mask in masks:
        # draw the mask on graph
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        show_mask(mask, plt.gca())
        plt.legend()
        plt.show()



    best_mask = masks[np.argmax(scores)]

    # Border detection
    detector = BorderDetector(best_mask)
    left_m, left_b = detector.find_left_border()
    top_m, top_b = detector.find_top_border()
    right_m, right_b = detector.find_right_border()
    bottom_m, bottom_b = detector.find_bottom_border()

    # Calculate intersections
    try:
        top_left = calculate_intersection(left_m, left_b, top_m, top_b)
        top_right = calculate_intersection(right_m, right_b, top_m, top_b)
        bottom_left = calculate_intersection(left_m, left_b, bottom_m, bottom_b)
        bottom_right = calculate_intersection(right_m, right_b, bottom_m, bottom_b)
    except ValueError as e:
        print(e)
        return

    # Visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    show_mask(best_mask, plt.gca())
    plt.scatter(*top_left, color='red', s=50, label="Top Left")
    plt.scatter(*top_right, color='green', s=50, label="Top Right")
    plt.scatter(*bottom_left, color='blue', s=50, label="Bottom Left")
    plt.scatter(*bottom_right, color='orange', s=50, label="Bottom Right")
    plt.legend()
    plt.show()

    while True:
        webcam = cv2.VideoCapture(0)
        ret, frame = webcam.read()
        warped_image = crop_view(frame, top_left, top_right, bottom_right, bottom_left)
        cv2.imwrite("rectified_image.jpg", warped_image)
        cv2.imshow("Rectified Image", warped_image)
        print("Frame captured")
        # Check for the 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
