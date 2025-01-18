import time
import cv2
import numpy as np

from utils.useTeachableMachine import CircleRecognition


def detect_circles_simple(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=30, minRadius=1, maxRadius=40)

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            return a, b, r
    return None, None, None

# not needed
def calculate_average_color(cell, x, y, r):
    mask = np.zeros(cell.shape[:2], dtype=np.uint8)
    inner_radius = int(r * 0.25)  # 50% of the circle radius
    cv2.circle(mask, (x, y), inner_radius, 255, -1)  # Create a filled circle mask

    # Calculate the mean color inside the mask
    mean_color = cv2.mean(cell, mask=mask)
    return tuple(map(int, mean_color[:3]))  # Return as (B, G, R)

def divide_picture_into_cells(image_path=None, columns=7, rows=6):
    if image_path is None:
        # get picture from webcam feed
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        image = frame

    else:
        # Load the image
        image = cv2.imread(image_path)  # Replace with your image path
        if image is None:
            raise ValueError("Image not found or invalid path.")

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Calculate the dimensions of each cell
    cell_width = width // columns
    cell_height = height // rows

    # Divide the image into 7 columns and 6 rows
    grid_cells = []
    for row in range(rows):  # 6 rows
        for col in range(columns):  # 7 columns
            # Define the coordinates for the current cell
            x_start = col * cell_width
            x_end = (col + 1) * cell_width
            y_start = row * cell_height
            y_end = (row + 1) * cell_height

            # Crop the cell from the image
            cell = image[y_start:y_end, x_start:x_end]
            grid_cells.append(cell)

            # Optional: Save or display each cell
            cell_filename = f'cells/cell_{row}_{col}.jpg'
            cv2.imwrite(cell_filename, cell)

def process_each_cell(image_path=None, columns=7, rows=6, force_image=None, color_player1=(88, 168, 55), color_player2=(213, 84, 89)):
    if force_image is not None:
        image = force_image
    elif image_path is None:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        image = frame
        cap.release()
    else:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or invalid path.")

    matrix_board = []

    image_w_overlay = image.copy()

    cd = CircleRecognition()

    height, width, _ = image.shape
    cell_width = width // columns
    cell_height = height // rows

    for row in range(rows):
        matrix_board_row = []
        for col in range(columns):
            x_start = col * cell_width
            x_end = (col + 1) * cell_width
            y_start = row * cell_height
            y_end = (row + 1) * cell_height

            cell = image[y_start:y_end, x_start:x_end]

            # save the cell to a file
            cell_filename = f'cells/cell_{row}_{col}.jpg'
            cv2.imwrite(cell_filename, cell)

            # draw the rectangle on the image
            cv2.rectangle(image_w_overlay, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)

            x, y, r = detect_circles_simple(cell)

            if x is not None and y is not None and r is not None:
                avg_color = calculate_average_color(cell, x, y, r)

                # Draw the detected circle and display the average color
                cv2.circle(image_w_overlay, (x_start + x, y_start + y), r, avg_color, 2)
                cv2.circle(image_w_overlay, (x_start + x, y_start + y), 1, (0, 0, 255), 2)

                cv2.putText(image_w_overlay, f"{avg_color}", (x_start + x - 20, y_start + y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            class_name, confidence_score = cd.predict(cell)
            class_name = class_name[1:].strip()
            if class_name == "None":
                matrix_board_row.append(0)
            elif class_name == "Red":
                matrix_board_row.append(1)
            elif class_name == "Green":
                matrix_board_row.append(2)
            else:
                print("Error in class name", class_name)

            cv2.putText(image_w_overlay, f"{class_name} {int(confidence_score * 100)}%",
                        (x_start + 5, y_start + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        matrix_board.append(matrix_board_row)


    pic_name = f'Processed_Cells/processed_block.jpg'
    cv2.imwrite(pic_name, image_w_overlay)
    print(f"Finished processing all cells. Saved to {pic_name}.")
    return matrix_board, image_w_overlay
