import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
import matplotlib.pyplot as plt

def extract_and_display_orb_features(image_path, max_features=128):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    
    if descriptors is not None:
        if descriptors.shape[0] > max_features:
            descriptors = descriptors[:max_features, :]
        elif descriptors.shape[0] < max_features:
            padding = np.zeros((max_features - descriptors.shape[0], descriptors.shape[1]))
            descriptors = np.vstack((descriptors, padding))
    
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    resized_image = cv2.resize(image_with_keypoints, (512, 512))
    cv2.imshow("ORB Features", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return descriptors.flatten()

# def extract_and_display_hog_features(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         print(f"Error: Unable to load image from {image_path}")
#         return None
#     image = cv2.resize(image, (256, 256))
#     blur_image = cv2.GaussianBlur(image, (11,11), 0)
#     features, hog_image = hog(blur_image, block_norm='L2-Hys', pixels_per_cell=(8, 8), visualize=True)
    
#     # resized_hog_image = cv2.resize(hog_image, (512, 512))
    
    
#     return features

def extract_and_display_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None

    # Ensure consistent preprocessing if necessary (omit if not needed)
    image = cv2.resize(image, (256, 256))
    blur_image = cv2.GaussianBlur(image, (11,11), 0) 
    
    features, hog_image = hog(blur_image, pixels_per_cell=(8, 8),  block_norm='L2-Hys', visualize=True,)
    
    print(f"HOG feature shape for {image_path}: {features.shape}")
    
    plt.imshow(hog_image)
    plt.title("HOG Features")
    plt.show()
    
    return features


def extract_and_display_lbp_features(image_path, radius=1, n_points=8):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    
    resized_lbp_image = cv2.resize(lbp, (512, 512))
    plt.imshow(resized_lbp_image, cmap='gray')
    plt.title("LBP Features")
    plt.show()
    
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return hist

def extract_and_display_gabor_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    gabor_features = []
    for theta in range(4):  # Gabor kernel orientation
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  # Scale
            for lamda in np.pi/4 * np.array([0.5, 1.0]):  # Wavelength
                kernel = cv2.getGaborKernel((21, 21), sigma, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
                filtered_img = cv2.filter2D(gray_image, cv2.CV_8UC3, kernel)
                gabor_features.append(filtered_img.mean())
                gabor_features.append(filtered_img.var())
    
    # Display the Gabor filtered image (only showing one example for visualization)
    resized_filtered_img = cv2.resize(filtered_img, (512, 512))
    plt.imshow(resized_filtered_img, cmap='gray')
    plt.title("Gabor Features")
    plt.show()
    
    return np.array(gabor_features)

def extract_and_display_hu_moments(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Since Hu Moments are not easily visualized, we just print them
    print("Hu Moments:")
    print(hu_moments)
    
    resized_image = cv2.resize(image, (512, 512))
    cv2.imshow("Original Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return hu_moments

def edges(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (3,3), 0)
    resized_image = cv2.resize(blur_image, (512, 512))
    edges = cv2.Canny(resized_image, 100, 200)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage for each feature extraction and display function
image_path = "data/Image_2000.jpg"

# Display Hu Moments (printed values)
edges(image_path)

# Display ORB features
extract_and_display_orb_features(image_path)

# Display HOG features
extract_and_display_hog_features(image_path)

# Display LBP features
extract_and_display_lbp_features(image_path)

# Display Gabor features
extract_and_display_gabor_features(image_path)

# Display Hu Moments (printed values)
extract_and_display_hu_moments(image_path)