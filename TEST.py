from utils.circle_detection_utils import divide_picture_into_cells, process_each_cell
from utils.useTeachableMachine import CircleRecognition


#divide_picture_into_cells("Images/rectified_image.jpg", 7, 6)


matrix = process_each_cell("Images/rectified_image.jpg", 7, 6)
