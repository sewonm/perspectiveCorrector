import cv2
import numpy as np

# Load the image
image = cv2.imread("/Users/sewonmyung/programming/machinelearningtests/distortedImage.jpg")

# Define points for perspective transform
src_points = np.float32([[100, 200], [400, 200], [100, 600], [400, 600]])
dst_points = np.float32([[0, 0], [500, 0], [0, 800], [500, 800]])

# Compute the perspective transform matrix
matrix = cv2.getPerspectiveTransform(src_points, dst_points)
result = cv2.warpPerspective(image, matrix, (1600, 1000))

# Display the result
cv2.imshow("Perspective Correction", result)
cv2.waitKey(0)
cv2.destroyAllWindows()