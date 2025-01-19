#from utils.circle_detection_utils import divide_picture_into_cells
from utils.connect4_utils import find_possible_position_of_next_move
#from utils.useTeachableMachine import CircleRecognition


#divide_picture_into_cells("Images/rectified_image.jpg", 7, 6)

"""
matrix, image = process_each_cell("Images/rectified_image.jpg", 7, 6)
for row in matrix:
    print(row)"""

matrix = [[0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0]]

find_possible_position_of_next_move(matrix)