import cv2
import numpy as np

def extract_and_display_orb_features(image_path, max_features=128):
    # Read the image from the path
    image = cv2.imread(image_path)
    
    # Handle if the image does not exist or is invalid
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create ORB detector
    orb = cv2.ORB_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    
    # Ensure descriptors have a consistent size (max_features)
    if descriptors is not None:
        if descriptors.shape[0] > max_features:
            descriptors = descriptors[:max_features, :]
        elif descriptors.shape[0] < max_features:
            padding = np.zeros((max_features - descriptors.shape[0], descriptors.shape[1]))
            descriptors = np.vstack((descriptors, padding))
    
    # Draw keypoints on the original image (for visualization)
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    resized_image = cv2.resize(image_with_keypoints, (512,512))
    # Display the image with keypoints
    cv2.imshow("ORB Features", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return descriptors.flatten()

for i in ra
# Example Usage
image_path = "data/Image_1.jpg"

# Extract ORB features and display image with keypoints
extract_and_display_orb_features(image_path)
