import cv2
import numpy as np

# Load the image
image = cv2.imread("/Users/sewonmyung/Library/Mobile Documents/com~apple~CloudDocs/trapezoid2.webp")
assert image is not None, "Image not found!"
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

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

# Order the detected points
src_points = order_points(doc_corners)
dst_points = np.float32([[0, 0], [800, 0], [800, 1000], [0, 1000]])

# Print the source points
print("Source Points (src_points):", src_points)

# Step 7: Compute the perspective transform matrix and apply the warp
matrix = cv2.getPerspectiveTransform(src_points, dst_points)
inverse_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
warped = cv2.warpPerspective(original, matrix, (3000, 3000))

# Create a copy of the warped image for drawing
drawing = warped.copy()
is_drawing = False
last_point = None

# Drawing callback function
def draw(event, x, y, flags, param):
    global is_drawing, last_point, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        last_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
        if last_point is not None:
            cv2.line(drawing, last_point, (x, y), (0, 0, 255), 3)
            last_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
        last_point = None

# Set up the drawing window
cv2.namedWindow("Draw on Warped Image")
cv2.setMouseCallback("Draw on Warped Image", draw)

while True:
    # Display the drawing image
    cv2.imshow("Draw on Warped Image", drawing)

    # Press 'r' to revert the drawing back to the original perspective
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        # Warp the drawing back to the original perspective
        reverted = cv2.warpPerspective(drawing, inverse_matrix, (original.shape[1], original.shape[0]))
        cv2.imshow("Reverted Drawing", reverted)

    # Press 's' to save the images
    elif key == ord('s'):
        cv2.imwrite("/Users/sewonmyung/Library/Mobile Documents/com~apple~CloudDocs/warped_image_with_drawing.jpg", drawing)
        cv2.imwrite("/Users/sewonmyung/Library/Mobile Documents/com~apple~CloudDocs/reverted_image.jpg", reverted)
        print("Images saved successfully!")

    # Press 'q' to quit
    elif key == ord('q'):
        break

cv2.destroyAllWindows()