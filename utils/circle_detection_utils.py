import time
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2

from utils.connect4_utils import find_possible_position_of_next_move
from utils.useTeachableMachine import CircleRecognition
import cv2
import numpy as np

from utils.useTeachableMachine import CircleRecognition

cv = CircleRecognition()

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

def process_each_cell_single_core_old_old(image_path=None, columns=7, rows=6, circle_detector=None, force_image=None, color_player1=(88, 168, 55), color_player2=(213, 84, 89)):
    DEBUG = True
    margin_of_similarity = 0.1

    if circle_detector is None:
        cd = CircleRecognition()
    else:
        cd = circle_detector

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

            # draw the rectangle on the image
            cv2.rectangle(image_w_overlay, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)

            # save the cell to a file
            cell_filename = f'cells/cell_{row}_{col}.jpg'
            try:
                old_state_of_cell = cv2.imread(cell_filename)
                difference = cv2.subtract(old_state_of_cell, cell)
                normalized_diff = difference.mean() / 255
                if normalized_diff < margin_of_similarity / 255:
                    print("Images are similar")
                    cv2.imwrite(cell_filename, cell)
                    print("Saved cell, but skipping analysis")
                    continue
                else:
                    cv2.imwrite(cell_filename, cell)
                    print("Images are different", normalized_diff)

            except:
                print("No file found")

            if DEBUG:
                cv2.imwrite(cell_filename, cell)

                x, y, r = detect_circles_simple(cell)

                if x is not None and y is not None and r is not None :
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

def process_each_cell_single_core_old(image_path=None, columns=7, rows=6, circle_detector=None, force_image=None,
                                  old_matrix=None, color_player1=(88, 168, 55), color_player2=(213, 84, 89)):
    DEBUG = True
    margin_of_similarity = 0.0  # 0.8?

    old_matrix = old_matrix if old_matrix is not None else np.zeros((rows, columns), dtype=int)
    possible_moves = find_possible_position_of_next_move(old_matrix)

    if circle_detector is None:
        cd = CircleRecognition()
    else:
        cd = circle_detector

    # Load the image (either from camera, path, or provided directly)
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

    # Save the full image once
    full_image_filename = "Images/full_board.jpg"
    try:
        old_state_of_board = cv2.imread(full_image_filename)
    except:
        old_state_of_board = None
    cv2.imwrite(full_image_filename, image)
    print(f"Full board image saved as {full_image_filename}")

    matrix_board = []
    image_w_overlay = image.copy()

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

            # Extract the cell directly using slicing
            cell = image[y_start:y_end, x_start:x_end]

            if old_state_of_board is not None:
                old_state_of_cell = old_state_of_board[y_start:y_end, x_start:x_end]

                difference = cv2.subtract(old_state_of_cell, cell)
                normalized_diff = difference.mean() / 255
                if normalized_diff < margin_of_similarity / 255:
                    print("Images are similar")
                    print("Skipping analysis of cell", row, col)
                    continue
                else:
                    print("Images are different", normalized_diff)
            else:
                normalized_diff = -1

            if DEBUG:
                # Draw the rectangle on the overlay image
                cv2.rectangle(image_w_overlay, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)

                x, y, r = detect_circles_simple(cell)

                if x is not None and y is not None and r is not None:
                    avg_color = calculate_average_color(cell, x, y, r)

                    # Draw the detected circle and display the average color
                    cv2.circle(image_w_overlay, (x_start + x, y_start + y), r, avg_color, 2)
                    cv2.circle(image_w_overlay, (x_start + x, y_start + y), 1, (0, 0, 255), 2)

                    cv2.putText(image_w_overlay, f"{avg_color}", (x_start + x - 20, y_start + y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # Process the cell (e.g., classify it)
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

            if normalized_diff >= 0:
                cv2.putText(image_w_overlay, f"{str(normalized_diff)[:4]}",
                            (x_start + 5, y_start + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        matrix_board.append(matrix_board_row)

    # Save the processed board overlay
    pic_name = "Processed_Cells/processed_block.jpg"
    cv2.imwrite(pic_name, image_w_overlay)
    print(f"Finished processing all cells. Saved to {pic_name}.")
    return matrix_board, image_w_overlay


def process_each_cell_single_core(image_path=None, columns=7, rows=6, circle_detector=None, force_image=None, old_matrix=None, color_player1=(88, 168, 55), color_player2=(213, 84, 89)):
    DEBUG = True
    margin_of_similarity = 0.0  # Adjust as needed for sensitivity

    # Initialize the old matrix if not provided
    old_matrix = old_matrix if old_matrix is not None else np.zeros((rows, columns), dtype=int)

    # Find the possible positions of the next moves
    possible_moves = find_possible_position_of_next_move(old_matrix)

    # Initialize the circle detector
    if circle_detector is None:
        cd = CircleRecognition()
    else:
        cd = circle_detector

    # Load the image
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

    # Save the full image once
    full_image_filename = "Images/full_board.jpg"
    try:
        old_state_of_board = cv2.imread(full_image_filename)
    except Exception:
        old_state_of_board = None
    cv2.imwrite(full_image_filename, image)
    print(f"Full board image saved as {full_image_filename}")

    # Initialize the matrix for the new board
    matrix_board = old_matrix.copy()
    image_w_overlay = image.copy()

    # Calculate cell dimensions
    height, width, _ = image.shape
    cell_width = width // columns
    cell_height = height // rows

    # Process each possible move
    for col in range(columns):
        row = possible_moves[col]
        if row < 0:  # Skip if no valid move for this column
            continue

        x_start = col * cell_width
        x_end = (col + 1) * cell_width
        y_start = row * cell_height
        y_end = (row + 1) * cell_height

        # Extract the cell directly using slicing
        cell = image[y_start:y_end, x_start:x_end]

        # Compare with the previous state of the board
        if old_state_of_board is not None:
            old_state_of_cell = old_state_of_board[y_start:y_end, x_start:x_end]
            difference = cv2.subtract(old_state_of_cell, cell)
            normalized_diff = difference.mean() / 255
            if normalized_diff < margin_of_similarity / 255:
                print(f"Cell ({row}, {col}) is unchanged. Skipping.")
                continue
            else:
                print(f"Cell ({row}, {col}) has changed. Difference: {normalized_diff:.4f}")
        else:
            normalized_diff = -1

        if DEBUG:
            # Draw the rectangle on the overlay image
            cv2.rectangle(image_w_overlay, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)

            x, y, r = detect_circles_simple(cell)

            if x is not None and y is not None and r is not None:
                avg_color = calculate_average_color(cell, x, y, r)

                # Draw the detected circle and display the average color
                cv2.circle(image_w_overlay, (x_start + x, y_start + y), r, avg_color, 2)
                cv2.circle(image_w_overlay, (x_start + x, y_start + y), 1, (0, 0, 255), 2)

                cv2.putText(image_w_overlay, f"{avg_color}", (x_start + x - 20, y_start + y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Classify the cell
        class_name, confidence_score = cd.predict(cell)
        class_name = class_name[1:].strip()
        if class_name == "None":
            matrix_board[row, col] = 0
        elif class_name == "Red":
            matrix_board[row, col] = 1
        elif class_name == "Green":
            matrix_board[row, col] = 2
        else:
            print(f"Error in class name: {class_name}")

        cv2.putText(image_w_overlay, f"{class_name} {int(confidence_score * 100)}%",
                    (x_start + 5, y_start + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if normalized_diff >= 0:
            cv2.putText(image_w_overlay, f"{str(normalized_diff)[:4]}",
                        (x_start + 5, y_start + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Save the processed board overlay
    pic_name = "Processed_Cells/processed_block.jpg"
    cv2.imwrite(pic_name, image_w_overlay)
    print(f"Finished processing all cells. Saved to {pic_name}.")
    return matrix_board, image_w_overlay

def process_batch(args):
    """Process a batch of cells."""
    batch, cell_width, cell_height, image, rows, cols = args
    results = []
    for row, col in batch:
        x_start = col * cell_width
        x_end = (col + 1) * cell_width
        y_start = row * cell_height
        y_end = (row + 1) * cell_height
        cell = image[y_start:y_end, x_start:x_end]

        result = {
            "row": row,
            "col": col,
            "class_name": "None",
            "confidence_score": 0,
            "circle": None,
            "avg_color": None,
        }

        # Detect Circle
        x, y, r = detect_circles_simple(cell)
        if x is not None and y is not None and r is not None:
            avg_color = calculate_average_color(cell, x, y, r)
            result["circle"] = (x + x_start, y + y_start, r)
            result["avg_color"] = avg_color

        # Predict Class
        class_name, confidence_score = cv.predict(cell)
        result["class_name"] = class_name[1:].strip()
        result["confidence_score"] = confidence_score

        results.append(result)
    return results

def process_each_cell_modular(image_path=None, columns=7, rows=6, num_threads=6, force_image=None):
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

    height, width, _ = image.shape
    cell_width = width // columns
    cell_height = height // rows
    image_w_overlay = image.copy()

    # Divide work into batches for threads
    all_cells = [(row, col) for row in range(rows) for col in range(columns)]
    batch_size = len(all_cells) // num_threads
    batches = [all_cells[i * batch_size:(i + 1) * batch_size] for i in range(num_threads)]
    if len(all_cells) % num_threads != 0:
        batches[-1].extend(all_cells[num_threads * batch_size:])  # Add remaining cells to the last batch

    # Threaded processing
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_batch, (batch, cell_width, cell_height, image, rows, columns))
            for batch in batches
        ]
        results = []
        for future in futures:
            results.extend(future.result())

    # Create matrix board and overlay
    matrix_board = np.zeros((rows, columns), dtype=int)
    for result in results:
        row, col = result["row"], result["col"]
        class_name = result["class_name"]
        confidence_score = result["confidence_score"]

        # Update matrix board
        if class_name == "None":
            matrix_board[row, col] = 0
        elif class_name == "Red":
            matrix_board[row, col] = 1
        elif class_name == "Green":
            matrix_board[row, col] = 2

        # Draw overlay
        if result["circle"]:
            x, y, r = result["circle"]
            avg_color = result["avg_color"]
            cv2.circle(image_w_overlay, (x, y), r, avg_color, 2)
            cv2.circle(image_w_overlay, (x, y), 1, (0, 0, 255), 2)
            cv2.putText(image_w_overlay, f"{avg_color}", (x - 20, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        cv2.putText(image_w_overlay, f"{class_name} {int(confidence_score * 100)}%",
                    (col * cell_width + 5, row * cell_height + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Save final image
    pic_name = f'Processed_Cells/processed_block.jpg'
    cv2.imwrite(pic_name, image_w_overlay)
    print(f"Finished processing all cells. Saved to {pic_name}.")
    return matrix_board, image_w_overlay

