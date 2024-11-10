import cv2
import numpy as np

# Load the image
image = cv2.imread("/Users/sewonmyung/Library/Mobile Documents/com~apple~CloudDocs/trapezoid2.webp")
assert image is not None, "Image not found!"

# Resize for easier handling (optional)
image = cv2.resize(image, (800, 600))

# Define source points (trapezoid corners)
src_points = np.float32([[100, 200], [700, 200], [50, 500], [750, 500]])

# Define destination points (rectangle corners)
dst_points = np.float32([[0, 0], [800, 0], [0, 600], [800, 600]])

# Get the perspective transformation matrix
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply the perspective warp
warped = cv2.warpPerspective(image, matrix, (800, 600))

# Display the results
cv2.imshow("Original Image", image)
cv2.imshow("Warped Image", warped)

# Wait for a key press and save images if 's' is pressed
key = cv2.waitKey(0)
if key == ord('s'):
    cv2.imwrite("/path/to/save/original_image.jpg", image)
    cv2.imwrite("/path/to/save/warped_image.jpg", warped)
    print("Images saved successfully!")

cv2.destroyAllWindows()