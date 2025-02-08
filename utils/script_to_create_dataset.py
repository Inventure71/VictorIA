import tkinter as tk

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ImagePointSelection import ImageClick
from utils.border_detection import BorderDetector
from utils.calculate_intersection import calculate_intersection
from utils.calibration_utils import show_mask, show_points
from utils.camera_calibration import crop_view
from utils.circle_detection_utils import divide_picture_into_cells
from utils.cv2_utils import display_mask_image_with_intersections
from utils.sam_model_handler import Sam2ModelHandler


def handle_points_labels(points, labels, path):
    global input_point, input_label, image_path
    input_point = points
    input_label = labels
    image_path = path
    print("Image path set to: ", image_path)


def main():
    global input_point, input_label, image_path

    """DEFAULT VARIABLES"""
    USE_WEBCAM = True
    DEBUG = False



    image_path = "Images/connect4_6.jpeg"

    """START OF MAIN"""
    # Load this at the start so it doesn't have to be loaded multiple times
    # Handle SAM model
    # config_path, checkpoint_path
    # /Users/inventure71/PycharmProjects/sam2/sam2/configs/sam2.1
    sam_handler = Sam2ModelHandler("configs/sam2.1/sam2.1_hiera_l.yaml", "../sam2/checkpoints/sam2.1_hiera_large.pt")

    while True:
        # Initialize tkinter for image selection
        root = tk.Tk()
        print("Starting with default image path: ", image_path)
        app = ImageClick(root, handle_points_labels, image_path, USE_WEBCAM=USE_WEBCAM)
        root.mainloop()
        print("Image path:", image_path)
        if not image_path:
            print("No image selected. Exiting.")
            return

        # open image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Set the image for the SAM model
        sam_handler.set_image(image)

        if DEBUG:
            # draw these points on the image
            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)
            show_points(input_point, input_label, plt.gca())
            plt.show()

        # Predict masks
        masks, scores, _ = sam_handler.predict(input_point, input_label)
        if DEBUG:
            for mask in masks:
                # draw the mask on graph
                plt.figure(figsize=(10, 10))
                plt.imshow(image_rgb)
                show_mask(mask, plt.gca())
                plt.legend()
                plt.show()

        best_mask = masks[np.argmax(scores)]

        # Border detection
        # POSSIBLE TO CONVERT ALL OF THIS IN ONE FUNCTION
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
            intersections = {
                "top_left": top_left,
                "top_right": top_right,
                "bottom_left": bottom_left,
                "bottom_right": bottom_right
            }

        except ValueError as e:
            print(e)
            return

        if display_mask_image_with_intersections(image, best_mask, intersections):
            break
        else:
            USE_WEBCAM = False

    if DEBUG:
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

    webcam = cv2.VideoCapture(0)

    import time

    ret, frame = webcam.read()
    warped_image = crop_view(frame, top_left, top_right, bottom_right, bottom_left)
    cv2.imwrite("Processed_Cells/warped_image.jpg", warped_image)
    time.sleep(0.5)
    divide_picture_into_cells(image_path="Processed_Cells/warped_image.jpg")


main()