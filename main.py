import time

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
from utils.detection_utils_v2 import process_each_cell_single_core, process_each_cell_multithreaded
from utils.cv2_utils import display_mask_image
from utils.useTeachableMachine import CircleRecognition


def handle_points_labels(points, labels, path):
    global input_point, input_label, image_path
    input_point = points
    input_label = labels
    image_path = path
    print("Image path set to: ", image_path)


def main():
    global  input_point, input_label, image_path

    """DEFAULT VARIABLES"""
    USE_WEBCAM = True
    DEBUG = False
    MULTY_TREAD = True
    image_path = "Images/connect4_6.jpeg"
    player1_color = (88, 168, 55)
    player2_color = (213, 84, 89)

    """START OF MAIN"""
    # Load this at the start so it doesn't have to be loaded multiple times
    # Handle SAM model
    sam_handler = SamModelHandler("Models/SAM Model Checkpoint.pth")

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
        if display_mask_image(image_rgb, best_mask): # here it displays the mask
            break
        else:
            USE_WEBCAM = False

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
    except ValueError as e:
        print(e)
        return

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
    cd = CircleRecognition()

    import time

    fps_start_time = time.time()
    frame_count = 0
    matrix = None

    while True:
        ret, frame = webcam.read()
        warped_image = crop_view(frame, top_left, top_right, bottom_right, bottom_left)
        if MULTY_TREAD:
            matrix, overlay = process_each_cell_multithreaded("Images/rectified_image.jpg", 7, 6,circle_detector=cd, old_matrix=matrix, force_image=warped_image)
        else:
            matrix, overlay = process_each_cell_single_core("Images/rectified_image.jpg", 7, 6, circle_detector=cd, old_matrix=matrix,
                                                            force_image=warped_image)
        print(matrix)
        # cv2.imwrite("rectified_image.jpg", warped_image)
        cv2.imshow("Rectified Image", overlay)
        print("Frame captured")

        frame_count += 1
        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        if time_diff >= 1:
            fps = frame_count / time_diff
            print(f"FPS: {fps:.2f}")
            fps_start_time = time.time()
            frame_count = 0

        # Check for the 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()

if __name__ == "__main__":
    main()
