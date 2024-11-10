import cv2

# Load the image
image = cv2.imread("/Users/sewonmyung/Downloads/poseML.jpg", cv2.IMREAD_GRAYSCALE)
# Apply Canny edge detection
edges = cv2.Canny(image, threshold1=50, threshold2=150)

# Display the edges
cv2.imshow("Edges", edges)
cv2.waitKey(0)