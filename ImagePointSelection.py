import tkinter as tk
from tkinter import messagebox

import cv2  # OpenCV for webcam functionality
import numpy as np
from PIL import Image, ImageTk


class ImageClick:
    def __init__(self, root, callback, image_path, USE_WEBCAM=False):
        self.root = root
        self.callback = callback
        self.USE_WEBCAM = USE_WEBCAM
        self.filepath = image_path
        self.root.title("Connect 4")

        # Create a main frame to hold canvas and buttons
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas to display image or webcam feed
        self.canvas = tk.Canvas(self.main_frame, bg="white")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Frame for buttons
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # List to store points and labels
        self.points = []
        self.labels = []

        # Initialize variables for webcam
        self.cap = None
        self.is_capturing = False
        self.frame = None
        self.photo = None

        if USE_WEBCAM:
            self.setup_webcam()
        else:
            self.load_image()

        # Bind mouse clicks to the canvas
        self.canvas.bind("<Button-1>", self.on_left_click)
        # Bind both right-click buttons for compatibility
        self.canvas.bind("<Button-2>", self.on_right_click)
        self.canvas.bind("<Button-3>", self.on_right_click)

        if USE_WEBCAM:
            self.capture_button = tk.Button(self.button_frame, text="Capture Image", command=self.capture_image)
            self.capture_button.pack(side=tk.LEFT, padx=5)

        self.end_button = tk.Button(self.button_frame, text="End Calibration", command=self.end_script)
        self.end_button.pack(side=tk.LEFT, padx=5)

    def setup_webcam(self):
        """Initialize webcam and start the preview."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error", "Cannot access the webcam.")
            self.root.destroy()
            return
        self.is_capturing = True
        self.show_webcam_preview()

    def show_webcam_preview(self):
        """Continuously capture frames from the webcam and display them."""
        if not self.is_capturing:
            return  # Stop updating if no longer capturing

        ret, frame = self.cap.read()
        if ret:
            # Convert to PIL for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preview_img = Image.fromarray(frame)

            # Resize the webcam feed to a smaller size so buttons remain visible
            max_w, max_h = 640, 480
            preview_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

            self.display_image = ImageTk.PhotoImage(preview_img)
            self.canvas.config(width=self.display_image.width(), height=self.display_image.height())

            # Clear previous drawings from canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)

        # Schedule the next frame update
        self.root.after(30, self.show_webcam_preview)

    def capture_image(self):
        """Capture the current frame from the webcam and proceed as with a loaded image."""
        if self.cap and self.is_capturing:
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)

            if ret:
                # Convert the frame to RGB and then to PIL Image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.original_image = Image.fromarray(frame)
                self.filepath = "Images/captured_image.jpg"
                self.original_image.save(self.filepath)

                self.orig_width, self.orig_height = self.original_image.size

                # Stop capturing and release the webcam
                self.is_capturing = False
                self.cap.release()

                # Process the captured image as if loaded from disk
                self.process_loaded_image()
            else:
                print("Failed to capture image from webcam.")
                messagebox.showerror("Capture Error", "Failed to capture image from webcam.")

    def load_image(self):
        """Load an image from a file (or a fixed path)."""
        try:
            self.original_image = Image.open(self.filepath)
        except FileNotFoundError:
            messagebox.showerror("File Not Found", f"Cannot find the image file: {self.filepath}")
            self.root.destroy()
            return

        self.orig_width, self.orig_height = self.original_image.size
        self.process_loaded_image()

    def process_loaded_image(self):
        """Process the loaded or captured image: resize and display on canvas."""
        # Clear any existing preview or drawings from the canvas
        self.canvas.delete("all")

        # Figure out the maximum display size
        max_width = self.root.winfo_screenwidth() - 150
        max_height = self.root.winfo_screenheight() - 250

        # Determine new size to maintain aspect ratio if needed
        aspect_ratio = self.orig_width / self.orig_height
        if self.orig_width > max_width or self.orig_height > max_height:
            if aspect_ratio > 1:  # landscape
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
            else:  # portrait
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
        else:
            new_width, new_height = self.orig_width, self.orig_height

        self.new_width, self.new_height = new_width, new_height

        # Compute the scaling factor (to convert canvas coords back to original coords)
        self.scale_x = self.orig_width / self.new_width
        self.scale_y = self.orig_height / self.new_height

        # Create a resized version for display
        resized_img = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.display_image = ImageTk.PhotoImage(resized_img)

        # Update canvas size to match resized image
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)

    def on_left_click(self, event):
        """Left click => label = 1"""
        if self.USE_WEBCAM and self.is_capturing:
            messagebox.showinfo("Info", "Please capture an image before selecting points.")
            return

        canvas_x, canvas_y = event.x, event.y
        # Convert canvas coords to original image coords
        orig_x = int(canvas_x * self.scale_x)
        orig_y = int(canvas_y * self.scale_y)

        print(f"[Left Click] Canvas coords: ({canvas_x}, {canvas_y}) -> Original coords: ({orig_x}, {orig_y})")
        self.points.append((orig_x, orig_y))
        self.labels.append(1)

        # Draw a red circle on the canvas
        self.canvas.create_oval(canvas_x - 5, canvas_y - 5,
                                canvas_x + 5, canvas_y + 5,
                                fill="red", outline="white", width=1)

    def on_right_click(self, event):
        """Right click => label = 0"""
        if self.USE_WEBCAM and self.is_capturing:
            messagebox.showinfo("Info", "Please capture an image before selecting points.")
            return

        canvas_x, canvas_y = event.x, event.y
        # Convert canvas coords to original image coords
        orig_x = int(canvas_x * self.scale_x)
        orig_y = int(canvas_y * self.scale_y)

        print(f"[Right Click] Canvas coords: ({canvas_x}, {canvas_y}) -> Original coords: ({orig_x}, {orig_y})")
        self.points.append((orig_x, orig_y))
        self.labels.append(0)

        # Draw a blue circle on the canvas
        self.canvas.create_oval(canvas_x - 5, canvas_y - 5,
                                canvas_x + 5, canvas_y + 5,
                                fill="blue", outline="white", width=1)

    def end_script(self):
        """
        When the user presses "End Script",
        convert lists to np.array and pass them to callback,
        then close the GUI.
        """
        if self.USE_WEBCAM and self.is_capturing:
            messagebox.showinfo("Info", "Please capture an image before ending the script.")
            return

        # Convert to NumPy arrays
        points_array = np.array(self.points)
        labels_array = np.array(self.labels)
        print("Final Points array:", points_array)
        print("Final Labels array:", labels_array)

        # Send to the callback
        self.callback(points_array, labels_array, self.filepath)

        # Cleanup resources
        if self.cap and self.cap.isOpened():
            self.cap.release()

        # Close the Tk window
        self.root.quit()
        self.root.destroy()
