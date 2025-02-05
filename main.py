import os
import random
import sys
import threading
import time

import cv2
import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib.pyplot as plt

from ImagePointSelection import ImageClick
from connect4 import predict
from utils.sam_model_handler import Sam2ModelHandler
from utils.calibration_utils import show_mask, show_points
from utils.calculate_intersection import calculate_intersection
from utils.border_detection import BorderDetector
from utils.camera_calibration import crop_view
from utils.detection_utils_v2 import process_each_cell_single_core, process_each_cell_multithreaded
from utils.cv2_utils import display_mask_image, display_mask_image_with_intersections
from utils.useTeachableMachine import CircleRecognition
import tensorflow as tf

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
    robot = 1 # 1 for player 1, 2 for player 2

    # Suppress TensorFlow debugging logs
    tf.keras.utils.disable_interactive_logging()

    image_path = "Images/connect4_6.jpeg"
    player1_color = (88, 168, 55)
    player2_color = (213, 84, 89)

    """START OF MAIN"""
    # Load this at the start so it doesn't have to be loaded multiple times
    # Handle SAM model
    #config_path, checkpoint_path
    #/Users/inventure71/PycharmProjects/sam2/sam2/configs/sam2.1
    sam_handler = Sam2ModelHandler("configs/sam2.1/sam2.1_hiera_l.yaml","../sam2/checkpoints/sam2.1_hiera_large.pt")

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
    cd = CircleRecognition()

    import time

    fps = 0
    fps_start_time = time.time()
    frame_count = 0
    matrix = np.zeros((6, 7), dtype=int)
    last_matrix = np.zeros((6, 7), dtype=int)

    player_that_needs_to_play = 1

    move, score = predict(matrix)
    print("Predicted move:", move)

    while True:
        frame_count += 1
        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        if time_diff >= 1:
            fps = frame_count / time_diff
            fps_start_time = time.time()
            frame_count = 0

        ret, frame = webcam.read()
        warped_image = crop_view(frame, top_left, top_right, bottom_right, bottom_left)
        if MULTY_TREAD:
            matrix, overlay = process_each_cell_multithreaded("Images/rectified_image.jpg", 7, 6, circle_detector=cd,
                                                              old_matrix=matrix, force_image=warped_image)
        else:
            matrix, overlay = process_each_cell_single_core("Images/rectified_image.jpg", 7, 6, circle_detector=cd,
                                                            old_matrix=matrix,
                                                            force_image=warped_image)

        new_matrix = matrix - last_matrix

        if new_matrix.max() == 1:
            player_that_needs_to_play = 2
            status_message = "Player 1 has played"
            # Replace this section in your loop
            print(f"\rFPS: {fps:.2f} | Matrix:\n{matrix} | Status: {status_message}", end='', flush=True)

        elif new_matrix.max() == 2:
            player_that_needs_to_play = 1
            move, best_score = predict(matrix)
            status_message = f"Player 2 has played | Predicted move: {move}"
            # Replace this section in your loop
            print(f"\rFPS: {fps:.2f} | Matrix:\n{matrix} | Status: {status_message}", end='', flush=True)

        else:
            if player_that_needs_to_play == 1:
                status_message = f"Move of player 1: {move}, waiting for it to be played"
                # Highlight the predicted move on the overlay
                column = move
                row = np.argmin(matrix[:, column])  # Find the lowest empty row in the column
                if row < 6:  # Ensure the row is within bounds
                    # Calculate the center of the cell
                    cell_height = overlay.shape[0] // 6
                    cell_width = overlay.shape[1] // 7
                    center_x = int(column * cell_width + cell_width / 2)
                    center_y = int(row * cell_height + cell_height / 2)

                    # Draw the circle on the overlay
                    cv2.circle(overlay, (center_x, center_y), radius=20, color=(0, 255, 0), thickness=2)
            elif player_that_needs_to_play == 2:
                status_message = f"Waiting for player 2 to play"

        last_matrix = matrix

        # Display matrix and overlay
        overlay = cv2.flip(overlay, 1)
        cv2.imshow("Rectified Image", overlay)

        # Check for the 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()

if __name__ == "__main__":
    main()
