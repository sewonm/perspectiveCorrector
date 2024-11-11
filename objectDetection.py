import cv2
import numpy as np

# Load YOLO model files
config_path = "yolov4.cfg"
weights_path = "yolov4.weights"
names_path = "coco.names"

# Load class names
with open(names_path, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Initialize the YOLO model
net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load an image
image_path = "/Users/sewonmyung/perspectiveCorrector/poseML.jpg"
image = cv2.imread(image_path)
height, width, _ = image.shape

# Prepare the image for the model
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Perform detection
detections = net.forward(output_layers)

# Process the results
confidence_threshold = 0.5
nms_threshold = 0.4

boxes = []
confidences = []
class_ids = []

for output in detections:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > confidence_threshold:
            # Get bounding box coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression (NMS)
indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

# Draw the bounding boxes on the image

for i in indices:
    # Retrieve the box, class ID, and confidence score
    box = boxes[i]
    x, y, w, h = box
    class_id = class_ids[i]
    confidence = confidences[i]
    class_name = class_names[class_id]

    # Print details to the console
    print(f"Detected Object: {class_name}")
    print(f"Confidence Score: {confidence:.2f}")
    print(f"Bounding Box: x={x}, y={y}, width={w}, height={h}")
    print("-" * 30)

    # Draw the bounding box and label on the image (optional)
    label = f"{class_name}: {confidence:.2f}"
    color = (0, 255, 0)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



# Display the output image
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
