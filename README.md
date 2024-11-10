Automatic Perspective Correction Tool

📸 Project Overview

This project leverages OpenCV to perform automatic perspective correction on images containing rectangular or trapezoidal objects. It detects the region of interest, such as documents, paintings, or signs, and applies a perspective transformation to warp the object into a rectangular shape. This process is useful for tasks like document scanning, image rectification, and preprocessing for computer vision applications.

✨ Key Features

	•	Automatic Detection: Identifies rectangular or trapezoidal regions using edge detection and contour analysis.
	•	Robust Corner Detection: Orders the detected corners correctly to ensure accurate perspective correction.
	•	Flexible Transformation: Dynamically adjusts the output size to preserve the aspect ratio of the detected object.
	•	User-Friendly Interface: Displays the original image with detected corners and the corrected, warped output image.
	•	Save Functionality: Allows users to save both the original and warped images with a single key press.

🔧 Technologies Used

	•	OpenCV: For image processing, edge detection, contour finding, and perspective transformation.
	•	NumPy: For efficient numerical operations and array handling.

💡 Use Cases

	•	Document Scanning: Automatically straighten and crop photos of documents or receipts.
	•	Art Rectification: Correct the perspective of photos taken at an angle, such as pictures of paintings or posters.
	•	Preprocessing for OCR: Prepare images for text extraction by rectifying the perspective distortion.