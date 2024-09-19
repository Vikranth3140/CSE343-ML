import cv2
import pandas as pd
import os
from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

labels_df = pd.read_csv('label.csv')

image_directory = 'data'

# Extract HOG features
def extract_hog_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (128, 128))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys',
                              visualize=True)
    return features

# Extract color histograms
def extract_color_histogram(image_path, bins=(8, 8, 8)):
    image = cv2.imread(image_path)
    if image is None:
        return None
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

hog_features_list = []
color_histogram_list = []

for index, row in labels_df.iterrows():
    image_path = os.path.join(image_directory, row['filename'])
    
    hog_features = extract_hog_features(image_path)
    if hog_features is not None:
        hog_features_list.append(hog_features)
    
    color_histogram = extract_color_histogram(image_path)
    if color_histogram is not None:
        color_histogram_list.append(color_histogram)

if len(hog_features_list) != len(color_histogram_list):
    raise ValueError("Mismatch between the number of HOG features and color histograms.")

# Combine HOG and Color Histogram features
combined_features = []
for hog, color_hist in zip(hog_features_list, color_histogram_list):
    combined_features.append(np.hstack((hog, color_hist)))

# Convert to NumPy array
X = np.array(combined_features)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction (optional, tune n_components)
pca = PCA(n_components=0.95)  # Keep 95% of the variance
X_pca = pca.fit_transform(X)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels_df['label'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Random Forest with more trees and tuned parameters
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Improved Random Forest Model Accuracy: {accuracy:.4f}")