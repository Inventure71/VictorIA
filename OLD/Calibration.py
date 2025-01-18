import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from ImagePointSelection import ImageClick
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from sympy.stats.sampling.sample_numpy import numpy
from scipy.stats import trim_mean
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import trim_mean


def handle_points_labels(points, labels):
    global input_point, input_label
    print("Points:", points)
    print("Labels:", labels)
    input_point = points
    input_label = labels
    #input_point = np.array([[480, 475]])
    #input_label = np.array([1])

root = tk.Tk()
app = ImageClick(root, handle_points_labels)
root.mainloop()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


if 'input_point' in globals() and 'input_label' in globals():
    # Load the original image with OpenCV for display
    image_bgr = cv2.imread("../connect4.jpeg")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    # 'input_point' is Nx2, 'input_label' is Nx(0 or 1)
    pos_points = input_point[input_label == 1]
    neg_points = input_point[input_label == 0]
    # Show them in green vs red
    plt.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=150, label='Left-click')
    plt.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=150, label='Right-click')
    plt.legend()
    plt.show()



import sys
sys.path.append("../..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "SAM Model Checkpoint.pth"
model_type = "vit_h"

device = "mps"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

image = cv2.imread('../connect4.jpeg')
predictor.set_image(image)


plt.figure(figsize=(10,10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

def find_left_border(mask):
    # Get the coordinates of all points in the mask where mask == 1
    y_coords, x_coords = np.where(mask == 1)

    # Combine the coordinates into a single array of shape (N, 2)
    mask_points = np.column_stack((x_coords, y_coords))

    # Sort the points by y coordinate in descending order, then by x coordinate in ascending order
    sorted_points = mask_points[np.lexsort((mask_points[:, 0], -mask_points[:, 1]))]

    # Iterate through the sorted points and find the leftmost point for each y coordinate
    leftmost_points = {}
    for x, y in sorted_points:
        if y not in leftmost_points:
            leftmost_points[y] = x

    print("Leftmost points for each y coordinate:")
    for y in sorted(leftmost_points.keys(), reverse=True):
        print(f"y = {y}, x = {leftmost_points[y]}")

    # Extract y and x arrays
    y = np.array(list(leftmost_points.keys()))
    x = np.array(list(leftmost_points.values()))

    # Calculate the 30% trimmed mean
    trimmed_mean_x = trim_mean(x, proportiontocut=0.4)

    # Calculate the thresholds for trimming
    lower_bound = np.percentile(x, 40)  # 30% trimmed: 15% from each side
    upper_bound = np.percentile(x, 60)

    # Identify points within and outside the trimmed range
    trimmed_indices = (x >= lower_bound) & (x <= upper_bound)
    trimmed_x = x[trimmed_indices]
    trimmed_y = y[trimmed_indices]
    outlier_x = x[~trimmed_indices]
    outlier_y = y[~trimmed_indices]

    # Linear fitting line using only trimmed data
    m, b = np.polyfit(trimmed_y, trimmed_x, 1)
    print(f"Linear fitting line (trimmed mean): x = {m:.2f}y + {b:.2f}")

    # Draw the results
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='viridis')
    plt.scatter(trimmed_x, trimmed_y, color='green', s=50, label="Trimmed points")
    plt.scatter(outlier_x, outlier_y, color='red', s=50, label="Outlier points")
    plt.axhline(trimmed_mean_x, color='orange', linestyle='--', label="30% Trimmed Mean")
    plt.plot(m * y + b, y, color='blue', linewidth=2, label="Linear fit (trimmed)")
    plt.legend()
    plt.show()
    return m, b
def find_bottom_border(mask):
    # Get the coordinates of all points in the mask where mask == 1
    y_coords, x_coords = np.where(mask == 1)

    # Combine the coordinates into a single array of shape (N, 2)
    mask_points = np.column_stack((x_coords, y_coords))

    # Sort the points by x coordinate in ascending order, then by y coordinate in descending order
    sorted_points = mask_points[np.lexsort((-mask_points[:, 1], mask_points[:, 0]))]

    # Iterate through the sorted points and find the lowest point for each x coordinate
    lowest_points = {}
    for x, y in sorted_points:
        if x not in lowest_points:
            lowest_points[x] = y

    print("Lowest points for each x coordinate:")
    for x in sorted(lowest_points.keys()):
        print(f"x = {x}, y = {lowest_points[y]}")

    # Extract x and y arrays
    x = np.array(list(lowest_points.keys()))
    y = np.array(list(lowest_points.values()))

    # Calculate the 30% trimmed mean
    trimmed_mean_y = trim_mean(y, proportiontocut=0.4)

    # Calculate the thresholds for trimming
    lower_bound = np.percentile(y, 40)  # 30% trimmed: 15% from each side
    upper_bound = np.percentile(y, 60)

    # Identify points within and outside the trimmed range
    trimmed_indices = (y >= lower_bound) & (y <= upper_bound)
    trimmed_x = x[trimmed_indices]
    trimmed_y = y[trimmed_indices]
    outlier_x = x[~trimmed_indices]
    outlier_y = y[~trimmed_indices]

    # Linear fitting line using only trimmed data
    m, b = np.polyfit(trimmed_x, trimmed_y, 1)
    print(f"Linear fitting line (trimmed mean): y = {m:.2f}x + {b:.2f}")

    # Draw the results
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='viridis')
    plt.scatter(trimmed_x, trimmed_y, color='green', s=50, label="Trimmed points")
    plt.scatter(outlier_x, outlier_y, color='red', s=50, label="Outlier points")
    plt.axhline(trimmed_mean_y, color='orange', linestyle='--', label="30% Trimmed Mean")
    plt.plot(x, m * x + b, color='blue', linewidth=2, label="Linear fit (trimmed)")
    plt.legend()
    plt.show()
    return m, b
def find_top_border(mask):
    # Get the coordinates of all points in the mask where mask == 1
    y_coords, x_coords = np.where(mask == 1)

    # Combine the coordinates into a single array of shape (N, 2)
    mask_points = np.column_stack((x_coords, y_coords))

    # Sort the points by x coordinate in ascending order, then by y coordinate in ascending order
    sorted_points = mask_points[np.lexsort((mask_points[:, 1], mask_points[:, 0]))]

    # Iterate through the sorted points and find the highest point for each x coordinate
    highest_points = {}
    for x, y in sorted_points:
        if x not in highest_points:
            highest_points[x] = y

    print("Highest points for each x coordinate:")
    for x in sorted(highest_points.keys()):
        print(f"x = {x}, y = {highest_points[x]}")

    # Extract x and y arrays
    x = np.array(list(highest_points.keys()))
    y = np.array(list(highest_points.values()))

    # Calculate the 30% trimmed mean
    trimmed_mean_y = trim_mean(y, proportiontocut=0.4) #0.3

    # Calculate the thresholds for trimming
    lower_bound = np.percentile(y, 40)
    upper_bound = np.percentile(y, 60)

    # Identify points within and outside the trimmed range
    trimmed_indices = (y >= lower_bound) & (y <= upper_bound)
    trimmed_x = x[trimmed_indices]
    trimmed_y = y[trimmed_indices]
    outlier_x = x[~trimmed_indices]
    outlier_y = y[~trimmed_indices]

    # Linear fitting line using only trimmed data
    m, b = np.polyfit(trimmed_x, trimmed_y, 1)
    print(f"Linear fitting line (trimmed mean): y = {m:.2f}x + {b:.2f}")

    # Draw the results
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='viridis')
    plt.scatter(trimmed_x, trimmed_y, color='green', s=50, label="Trimmed points")
    plt.scatter(outlier_x, outlier_y, color='red', s=50, label="Outlier points")
    plt.axhline(trimmed_mean_y, color='orange', linestyle='--', label="30% Trimmed Mean")
    plt.plot(x, m * x + b, color='blue', linewidth=2, label="Linear fit (trimmed)")
    plt.legend()
    plt.show()
    return m, b
def find_right_border(mask):
    # Get the coordinates of all points in the mask where mask == 1
    y_coords, x_coords = np.where(mask == 1)

    # Combine the coordinates into a single array of shape (N, 2)
    mask_points = np.column_stack((x_coords, y_coords))

    # Sort the points by y coordinate in descending order, then by x coordinate in descending order
    sorted_points = mask_points[np.lexsort((-mask_points[:, 0], -mask_points[:, 1]))]

    # Iterate through the sorted points and find the rightmost point for each y coordinate
    rightmost_points = {}
    for x, y in sorted_points:
        if y not in rightmost_points:
            rightmost_points[y] = x

    print("Rightmost points for each y coordinate:")
    for y in sorted(rightmost_points.keys(), reverse=True):
        print(f"y = {y}, x = {rightmost_points[y]}")

    # Extract y and x arrays
    y = np.array(list(rightmost_points.keys()))
    x = np.array(list(rightmost_points.values()))

    # Calculate the 30% trimmed mean
    trimmed_mean_x = trim_mean(x, proportiontocut=0.4)

    # Calculate the thresholds for trimming
    lower_bound = np.percentile(x, 40) #15
    upper_bound = np.percentile(x, 60) #85

    # Identify points within and outside the trimmed range
    trimmed_indices = (x >= lower_bound) & (x <= upper_bound)
    trimmed_x = x[trimmed_indices]
    trimmed_y = y[trimmed_indices]
    outlier_x = x[~trimmed_indices]
    outlier_y = y[~trimmed_indices]

    # Linear fitting line using only trimmed data
    m, b = np.polyfit(trimmed_y, trimmed_x, 1)
    print(f"Linear fitting line (trimmed mean): x = {m:.2f}y + {b:.2f}")

    # Draw the results
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='viridis')
    plt.scatter(trimmed_x, trimmed_y, color='green', s=50, label="Trimmed points")
    plt.scatter(outlier_x, outlier_y, color='red', s=50, label="Outlier points")
    plt.axhline(trimmed_mean_x, color='orange', linestyle='--', label="30% Trimmed Mean")
    plt.plot(m * y + b, y, color='blue', linewidth=2, label="Linear fit (trimmed)")
    plt.legend()
    plt.show()
    return m, b

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()

best_mask = masks[np.argmax(scores)]
for side in ['left', 'right', 'top', 'bottom']:
    print(f"Processing {side} border...")

# Define helper functions for intersection calculations
def calculate_intersection_left_top(m_left, b_left, m_top, b_top):
    denominator = 1 - m_top * m_left
    if denominator == 0:
        raise ValueError("Left and Top borders are parallel and do not intersect.")
    y = (m_top * b_left + b_top) / denominator
    x = m_left * y + b_left
    return x, y

def calculate_intersection_right_top(m_right, b_right, m_top, b_top):
    denominator = 1 - m_top * m_right
    if denominator == 0:
        raise ValueError("Right and Top borders are parallel and do not intersect.")
    y = (m_top * b_right + b_top) / denominator
    x = m_right * y + b_right
    return x, y

def calculate_intersection_left_bottom(m_left, b_left, m_bottom, b_bottom):
    denominator = 1 - m_bottom * m_left
    if denominator == 0:
        raise ValueError("Left and Bottom borders are parallel and do not intersect.")
    y = (m_bottom * b_left + b_bottom) / denominator
    x = m_left * y + b_left
    return x, y

def calculate_intersection_right_bottom(m_right, b_right, m_bottom, b_bottom):
    denominator = 1 - m_bottom * m_right
    if denominator == 0:
        raise ValueError("Right and Bottom borders are parallel and do not intersect.")
    y = (m_bottom * b_right + b_bottom) / denominator
    x = m_right * y + b_right
    return x, y

# Process the best mask
best_mask = masks[np.argmax(scores)]
for side in ['left', 'right', 'top', 'bottom']:
    print(f"Processing {side} border...")

left_m, left_b = find_left_border(best_mask)
top_m, top_b = find_top_border(best_mask)
right_m, right_b = find_right_border(best_mask)
bottom_m, bottom_b = find_bottom_border(best_mask)

# Calculate the intersection points correctly
try:
    top_left = calculate_intersection_left_top(left_m, left_b, top_m, top_b)
    top_right = calculate_intersection_right_top(right_m, right_b, top_m, top_b)
    bottom_left = calculate_intersection_left_bottom(left_m, left_b, bottom_m, bottom_b)
    bottom_right = calculate_intersection_right_bottom(right_m, right_b, bottom_m, bottom_b)
except ValueError as e:
    print(e)
    # Handle parallel lines or other exceptions as needed

# Draw the results
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(mask, plt.gca())
plt.scatter(*top_left, color='red', s=50, label="Top Left")
plt.scatter(*top_right, color='green', s=50, label="Top Right")
plt.scatter(*bottom_left, color='blue', s=50, label="Bottom Left")
plt.scatter(*bottom_right, color='orange', s=50, label="Bottom Right")
plt.legend()
plt.show()


