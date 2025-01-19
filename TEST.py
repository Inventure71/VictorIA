#from utils.circle_detection_utils import divide_picture_into_cells
import sys
import time

import numpy as np

#from connect4 import predict_move
#from utils.connect4_utils import find_possible_position_of_next_move
#from utils.useTeachableMachine import CircleRecognition
#import tensorflow as tf

#divide_picture_into_cells("Images/rectified_image.jpg", 7, 6)

"""
matrix, image = process_each_cell("Images/rectified_image.jpg", 7, 6)
for row in matrix:
    print(row)"""

matrix = np.matrix(([0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 1, 2, 0]))

last_matrix = np.matrix(([0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 2, 0]))

new_matrix = matrix-last_matrix

print(new_matrix.mean()*6*7)
print(new_matrix.max())
print(new_matrix)


while True:
    fps =+ 1
    status_message = "Waiting for input"

    import sys

    # Replace this section in your loop
    print(f"\rFPS: {fps:.2f} | Matrix:\n{matrix} | Status: {status_message}", end='', flush=True)

    time.sleep(1)


"""
matrix = numpy.array(matrix)

model = tf.keras.models.load_model("connect4_model_v1.h5")
print(predict_move(model, matrix))"""