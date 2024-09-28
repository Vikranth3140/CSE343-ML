import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA

# Load the labels
labels_df = pd.read_csv('label.csv')

image_directory = 'data'

# Define the PCA model for edges
pca = PCA(n_components=50)  # Adjust the number of components as needed

# Function to extract edges and apply PCA
def extract_edges_pca(image_path, pca_model=None):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (11, 11), 0)
    edges = cv2.Canny(blur_image, 100, 200).flatten()
    
    if pca_model is not None:
        edges = pca_model.transform([edges])[0]  # Apply PCA
    
    return edges

# Function to extract ORB features
def extract_orb_features(image_path, max_features=128):
    image = cv2.imread(image_path)
    if image is None:
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
        return descriptors.flatten()
    else:
        return np.zeros(max_features * 32)

# Function to extract HOG features
def extract_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (256, 256))
    blur_image = cv2.GaussianBlur(image, (11,11), 0)
    features, _ = hog(blur_image, pixels_per_cell=(8, 8), block_norm='L2-Hys', visualize=False)
    
    return features

# Function to extract LBP features
def extract_lbp_features(image_path, radius=1, n_points=8):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# Function to extract color histogram features
def extract_color_histogram(image_path, bins=(8, 8, 8)):
    image = cv2.imread(image_path)
    if image is None:
        return None
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Combine all features into a single feature vector
def extract_combined_features(image_path, pca_model):
    edges_pca = extract_edges_pca(image_path, pca_model=pca_model)
    orb_features = extract_orb_features(image_path)
    hog_features = extract_hog_features(image_path)
    lbp_features = extract_lbp_features(image_path)
    color_histogram = extract_color_histogram(image_path)

    feature_list = [edges_pca, orb_features, hog_features, lbp_features, color_histogram]
    max_length = max(len(f) for f in feature_list if f is not None)

    padded_features = []
    for feature in feature_list:
        if feature is not None:
            if len(feature) < max_length:
                feature = np.pad(feature, (0, max_length - len(feature)), mode='constant')
            padded_features.append(feature)

    combined_features = np.hstack(padded_features)
    return combined_features

# Process images in batches
batch_size = 1000  # Adjust batch size according to your system’s memory
num_images = len(labels_df)
num_batches = num_images // batch_size + (num_images % batch_size != 0)

# First pass: fit PCA on edges
all_edges = []

for batch_num in range(num_batches):
    start_index = batch_num * batch_size
    end_index = min((batch_num + 1) * batch_size, num_images)

    for index, row in labels_df.iloc[start_index:end_index].iterrows():
        image_path = os.path.join(image_directory, row['filename'])
        edges = extract_edges_pca(image_path)  # Extract edges without PCA
        if edges is not None:
            all_edges.append(edges)

# Fit PCA on the collected edges
all_edges = np.array(all_edges)
pca.fit(all_edges)

# Second pass: extract features using PCA on edges
for batch_num in range(num_batches):
    start_index = batch_num * batch_size
    end_index = min((batch_num + 1) * batch_size, num_images)

    features_list = []
    labels = []
    file_names = []

    for index, row in labels_df.iloc[start_index:end_index].iterrows():
        image_path = os.path.join(image_directory, row['filename'])
        combined_features = extract_combined_features(image_path, pca_model=pca)
        if combined_features is not None:
            features_list.append(combined_features)
            labels.append(row['label'])
            file_names.append(row['filename'])

    # Convert to DataFrame
    batch_df = pd.DataFrame(features_list)
    batch_df['label'] = labels
    batch_df['filename'] = file_names

    # Append to the CSV file
    if batch_num == 0:
        batch_df.to_csv('extracted_features_pca.csv', index=False, mode='w')
    else:
        batch_df.to_csv('extracted_features_pca.csv', index=False, header=False, mode='a')

    print(f"Processed batch {batch_num + 1} of {num_batches}")