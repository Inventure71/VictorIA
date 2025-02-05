import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import trim_mean


class BorderDetector:
    def __init__(self, mask):
        self.mask = mask

    # REWRITE IN MORE EFFICIENT WAY

    def find_left_border(self):
        # Get the coordinates of all points in the mask where mask == 1
        y_coords, x_coords = np.where(self.mask == 1)

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
        trimmed_mean_x = trim_mean(x, proportiontocut=0.35)

        # Calculate the thresholds for trimming
        lower_bound = np.percentile(x, 35)  # 30% trimmed: 15% from each side
        upper_bound = np.percentile(x, 65)

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
        plt.imshow(self.mask, cmap='viridis')
        plt.scatter(trimmed_x, trimmed_y, color='green', s=50, label="Trimmed points")
        plt.scatter(outlier_x, outlier_y, color='red', s=50, label="Outlier points")
        plt.axhline(trimmed_mean_x, color='orange', linestyle='--', label="30% Trimmed Mean")
        plt.plot(m * y + b, y, color='blue', linewidth=2, label="Linear fit (trimmed)")
        plt.legend()
        plt.show()
        return m, b

    def find_bottom_border(self):
        # Get the coordinates of all points in the mask where mask == 1
        y_coords, x_coords = np.where(self.mask == 1)

        # Combine the coordinates into a single array of shape (N, 2)
        mask_points = np.column_stack((x_coords, y_coords))

        # Sort the points by x coordinate in ascending order, then by y coordinate in descending order
        sorted_points = mask_points[np.lexsort((-mask_points[:, 1], mask_points[:, 0]))]

        # Iterate through the sorted points and find the lowest point for each x coordinate
        lowest_points = {}
        for x, y in sorted_points:
            if x not in lowest_points:
                lowest_points[x] = y

        # Extract x and y arrays
        x = np.array(list(lowest_points.keys()))
        y = np.array(list(lowest_points.values()))

        # Calculate the 30% trimmed mean
        trimmed_mean_y = trim_mean(y, proportiontocut=0.35)

        # Calculate the thresholds for trimming
        lower_bound = np.percentile(y, 35)  # 30% trimmed: 15% from each side
        upper_bound = np.percentile(y, 65)

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
        plt.imshow(self.mask, cmap='viridis')
        plt.scatter(trimmed_x, trimmed_y, color='green', s=50, label="Trimmed points")
        plt.scatter(outlier_x, outlier_y, color='red', s=50, label="Outlier points")
        plt.axhline(trimmed_mean_y, color='orange', linestyle='--', label="30% Trimmed Mean")
        plt.plot(x, m * x + b, color='blue', linewidth=2, label="Linear fit (trimmed)")
        plt.legend()
        plt.show()
        return m, b

    def find_top_border(self):
        # Get the coordinates of all points in the mask where mask == 1
        y_coords, x_coords = np.where(self.mask == 1)

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
        trimmed_mean_y = trim_mean(y, proportiontocut=0.35)  # 0.3

        # Calculate the thresholds for trimming
        lower_bound = np.percentile(y, 35)
        upper_bound = np.percentile(y, 65)

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
        plt.imshow(self.mask, cmap='viridis')
        plt.scatter(trimmed_x, trimmed_y, color='green', s=50, label="Trimmed points")
        plt.scatter(outlier_x, outlier_y, color='red', s=50, label="Outlier points")
        plt.axhline(trimmed_mean_y, color='orange', linestyle='--', label="30% Trimmed Mean")
        plt.plot(x, m * x + b, color='blue', linewidth=2, label="Linear fit (trimmed)")
        plt.legend()
        plt.show()
        return m, b

    def find_right_border(self):
        # Get the coordinates of all points in the mask where mask == 1
        y_coords, x_coords = np.where(self.mask == 1)

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
        trimmed_mean_x = trim_mean(x, proportiontocut=0.35)

        # Calculate the thresholds for trimming
        lower_bound = np.percentile(x, 35)  # 15
        upper_bound = np.percentile(x, 65)  # 85

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
        plt.imshow(self.mask, cmap='viridis')
        plt.scatter(trimmed_x, trimmed_y, color='green', s=50, label="Trimmed points")
        plt.scatter(outlier_x, outlier_y, color='red', s=50, label="Outlier points")
        plt.axhline(trimmed_mean_x, color='orange', linestyle='--', label="30% Trimmed Mean")
        plt.plot(m * y + b, y, color='blue', linewidth=2, label="Linear fit (trimmed)")
        plt.legend()
        plt.show()
        return m, b
