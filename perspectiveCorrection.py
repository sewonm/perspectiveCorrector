import cv2
import numpy as np

# Initialize a list to store points
points = []

# Mouse callback function to capture points
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Points", image)

# Load the image
image = cv2.imread("/Users/sewonmyung/Library/Mobile Documents/com~apple~CloudDocs/distortedImage.jpg")
image = cv2.resize(image, (800, 600))

# Display the image and set the mouse callback function
cv2.imshow("Select Points", image)
cv2.setMouseCallback("Select Points", select_points)

# Wait until 4 points are selected
cv2.waitKey(0)

# Ensure 4 points are selected
if len(points) == 4:
    # Sort the points: top-left, top-right, bottom-left, bottom-right
    points = sorted(points, key=lambda x: x[1])  # Sort by y-coordinate
    if points[0][0] < points[1][0]:
        top_left, top_right = points[0], points[1]
    else:
        top_left, top_right = points[1], points[0]

    if points[2][0] < points[3][0]:
        bottom_left, bottom_right = points[2], points[3]
    else:
        bottom_left, bottom_right = points[3], points[2]

    src_points = np.float32([top_left, top_right, bottom_left, bottom_right])
    # Define the destination points based on the aspect ratio
    width = 800
    height = 1000
    dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    # Display the warped image
    cv2.imshow("Warped Image", warped)

    # Save the images if '1' is pressed
    key = cv2.waitKey(0)
    if key == ord('1'):
        cv2.imwrite("/Users/sewonmyung/programming/warpTestsImages/selected_points_image_corrected.jpg", image)
        cv2.imwrite("/Users/sewonmyung/programming/warpTestsImages/warped_image_corrected.jpg", warped)
        print("Images saved successfully!")

    cv2.destroyAllWindows()
else:
    print("Error: Please select exactly 4 points.")