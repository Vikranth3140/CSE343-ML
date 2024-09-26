import cv2
import pandas as pd
import os
from skimage.feature import hog
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# Load the labels
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

# Function to extract color histograms
def extract_color_histogram(image_path, bins=(8, 8, 8)):
    image = cv2.imread(image_path)
    if image is None:
        return None
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Extract HOG features and color histograms
hog_features_list = []
color_histogram_list = []
file_names = []

for index, row in labels_df.iterrows():
    image_path = os.path.join(image_directory, row['filename'])
    
    # Extract HOG features
    hog_features = extract_hog_features(image_path)
    if hog_features is not None:
        hog_features_list.append(hog_features)
    
    # Extract color histograms
    color_histogram = extract_color_histogram(image_path)
    if color_histogram is not None:
        color_histogram_list.append(color_histogram)
    
    # Store file name for potential misclassification analysis
    file_names.append(row['filename'])

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

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels_df['label'])

# Train-test split
X_train, X_test, y_train, y_test, file_train, file_test = train_test_split(
    X_pca, y, file_names, test_size=0.2, random_state=42)

# Initialize the Random Forest model with Grid Search for hyperparameter tuning
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search
best_rf = grid_search.best_estimator_

# Test the best model
y_pred = best_rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Random Forest Model Accuracy: {accuracy:.4f}")

# Identify misclassified images
misclassified_indices = np.where(y_pred != y_test)[0]

print("Misclassified Images:")
for index in misclassified_indices:
    print(f"File: {file_test[index]}, Predicted: {label_encoder.inverse_transform([y_pred[index]])[0]}, Actual: {label_encoder.inverse_transform([y_test[index]])[0]}")
