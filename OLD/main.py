import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from ImagePointSelection import ImageClick
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from sympy.stats.sampling.sample_numpy import numpy

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


# For example, show them on the image with matplotlib:
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

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()


