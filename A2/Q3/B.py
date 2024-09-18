import cv2
from skimage.feature import hog
import pandas as pd
import numpy as np
import os
from PIL import Image

labels_df = pd.read_csv('label.csv')

image_directory = 'data'

# Extract HOG features
def extract_hog_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    # Resize the image to a smaller fixed size
    image = cv2.resize(image, (128, 128))  # resizing for consistency
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Extract HOG features
    features, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys',
                              visualize=True)
    return features

# Testing the function
image_path = 'data/Image_6.jpg'
features = extract_hog_features(image_path)
if features is not None:
    print("HOG feature extraction successful!")
else:
    print("HOG feature extraction failed.")






# Function to extract color histograms
def extract_color_histogram(image_path, bins=(8, 8, 8)):
    image = cv2.imread(image_path)
    if image is None:
        return None
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Compute the color histogram in the HSV color space
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Extract Color Histogram features from each image and store them in a list
color_histogram_features_list = []

for index, row in labels_df.iterrows():
    image_path = os.path.join(image_directory, row['filename'])
    features = extract_color_histogram(image_path)
    if features is not None:
        color_histogram_features_list.append(features)

# Convert the color histogram features list to a DataFrame for analysis
color_histogram_df = pd.DataFrame(color_histogram_features_list)

print("Color Histogram feature extraction complete!")
print(color_histogram_df.head())  # Display first few extracted features