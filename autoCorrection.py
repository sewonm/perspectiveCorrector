import cv2
import numpy as np

# Load the image
image = cv2.imread("/Users/sewonmyung/Library/Mobile Documents/com~apple~CloudDocs/trapezoid2.webp")
original = image.copy()
image = cv2.resize(image, (800, 600))

# Step 1: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 3: Edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Step 4: Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 5: Sort contours by area and find the largest rectangle
contours = sorted(contours, key=cv2.contourArea, reverse=True)
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # If the contour has 4 points, we assume it’s a rectangle
    if len(approx) == 4:
        doc_corners = approx
        break

# Step 6: Order the corners correctly (top-left, top-right, bottom-right, bottom-left)
def order_points(pts):
    pts = pts.reshape((4, 2))
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of points (top-left has the smallest sum, bottom-right has the largest sum)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference of points (top-right has the smallest difference, bottom-left has the largest difference)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

# Order the detected points
src_points = order_points(doc_corners)
dst_points = np.float32([[0, 0], [800, 0], [800, 1000], [0, 1000]])

# Step 7: Compute the perspective transform matrix and apply the warp
matrix = cv2.getPerspectiveTransform(src_points, dst_points)
warped = cv2.warpPerspective(original, matrix, (1000, 2000))

# Step 8: Draw the detected corners on the original image
for point in src_points:
    cv2.circle(image, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)

# Step 9: Display the results
cv2.imshow("Original Image with Detected Corners", image)
cv2.imshow("Warped Image", warped)

# Step 10: Save the images if '1' is pressed
key = cv2.waitKey(0)
if key == ord('1'):
    cv2.imwrite("/path/to/save/original_with_corners.jpg", image)
    cv2.imwrite("/path/to/save/warped_image.jpg", warped)
    print("Images saved successfully!")

cv2.destroyAllWindows()