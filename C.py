import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import os
import pickle

# Load the labels
labels_df = pd.read_csv('label.csv')

image_directory = 'data'

def extract_edges(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (11, 11), 0)
    edges = cv2.Canny(blur_image, 100, 200)
    return edges.flatten()

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

def extract_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (256, 256))
    blur_image = cv2.GaussianBlur(image, (11,11), 0)
    features, _ = hog(blur_image, pixels_per_cell=(8, 8), block_norm='L2-Hys', visualize=True)
    
    # Compute mean, median, variance of HOG features
    mean_hog = np.mean(features)
    median_hog = np.median(features)
    var_hog = np.var(features)
    
    # Add these statistics to the HOG features
    hog_stats = np.array([mean_hog, median_hog, var_hog])
    combined_hog_features = np.concatenate([features, hog_stats])
    
    return combined_hog_features

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

def extract_gabor_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gabor_features = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.pi / 4 * np.array([0.5, 1.0]):
                kernel = cv2.getGaborKernel((21, 21), sigma, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
                filtered_img = cv2.filter2D(gray_image, cv2.CV_8UC3, kernel)
                gabor_features.append(filtered_img.mean())
                gabor_features.append(filtered_img.var())
    return np.array(gabor_features)

def extract_color_histogram(image_path, bins=(8, 8, 8)):
    image = cv2.imread(image_path)
    if image is None:
        return None
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Combine all features into a single feature vector
def extract_combined_features(image_path):
    edges = extract_edges(image_path)
    orb_features = extract_orb_features(image_path)
    hog_features = extract_hog_features(image_path)
    lbp_features = extract_lbp_features(image_path)
    gabor_features = extract_gabor_features(image_path)
    color_histogram = extract_color_histogram(image_path)

    # Ensure all features have a consistent length
    feature_list = [edges, orb_features, hog_features, lbp_features, gabor_features, color_histogram]
    max_length = max(len(f) for f in feature_list if f is not None)

    padded_features = []
    for feature in feature_list:
        if feature is not None:
            if len(feature) < max_length:
                feature = np.pad(feature, (0, max_length - len(feature)), mode='constant')
            padded_features.append(feature)

    combined_features = np.hstack(padded_features)
    return combined_features

# Extract features from all images
features_list = []
file_names = []

for index, row in labels_df.iterrows():
    image_path = os.path.join(image_directory, row['filename'])
    
    # Extract combined features
    combined_features = extract_combined_features(image_path)
    if combined_features is not None:
        features_list.append(combined_features)
        file_names.append(row['filename'])

# Convert to NumPy array
X = np.array(features_list)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels_df['label'])

# Train-test split
# First, select 6000 samples with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Initialize and train the Random Forest model with the provided best parameters
best_rf_params = {
    'max_depth': 25,
    'min_samples_leaf': 2,
    'min_samples_split': 5,
    'n_estimators': 1000
}

rf = RandomForestClassifier(**best_rf_params, random_state=42)
rf.fit(X_train, y_train)

# Test the model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy: {accuracy:.4f}")

# Optional: Save the model to a file
model_filename = 'trained_random_forest_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(rf, file)

# Identify misclassified images
misclassified_indices = np.where(y_pred != y_test)[0]
print("Misclassified Images:")
for index in misclassified_indices:
    print(f"File: {file_names[index]}, Predicted: {label_encoder.inverse_transform([y_pred[index]])[0]}, Actual: {label_encoder.inverse_transform([y_test[index]])[0]}")