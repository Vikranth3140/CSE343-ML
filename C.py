import cv2
import pandas as pd
import os
from skimage.feature import hog, local_binary_pattern
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import pickle

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

# Extract color histograms
def extract_color_histogram(image_path, bins=(8, 8, 8)):
    image = cv2.imread(image_path)
    if image is None:
        return None
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Extract LBP features
def extract_lbp_features(image_path, radius=1, n_points=8):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize the histogram
    return hist

# Extract Gabor features
def extract_gabor_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gabor_features = []
    for theta in range(4):  # Gabor kernel orientation
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  # Scale
            for lamda in np.pi/4 * np.array([0.5, 1.0]):  # Wavelength
                kernel = cv2.getGaborKernel((21, 21), sigma, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
                fimg = cv2.filter2D(gray_image, cv2.CV_8UC3, kernel)
                gabor_features.append(fimg.mean())
                gabor_features.append(fimg.var())
    return np.array(gabor_features)

# Extract Hu moments
def extract_hu_moments(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

# Extract ORB features
def extract_orb_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    if descriptors is not None:
        return descriptors.flatten()
    else:
        return np.zeros(128)  # Return a zero array if no keypoints are detected

# Combine all features into a single feature vector
def extract_combined_features(image_path):
    hog_features = extract_hog_features(image_path)
    color_histogram = extract_color_histogram(image_path)
    lbp_features = extract_lbp_features(image_path)
    gabor_features = extract_gabor_features(image_path)
    hu_moments = extract_hu_moments(image_path)
    orb_features = extract_orb_features(image_path)

    combined_features = np.hstack((
        hog_features,
        color_histogram,
        lbp_features,
        gabor_features,
        hu_moments,
        orb_features
    ))

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

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.80, random_state=42)
X_pca = pca.fit_transform(X)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels_df['label'])

# Train-test split
X_train, X_test, y_train, y_test, file_train, file_test = train_test_split(
    X_pca, y, file_names, test_size=0.2, random_state=42)

# Initialize the Random Forest model with Grid Search for hyperparameter tuning
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True]
}

grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)

# Initialize the XGBoost model with Grid Search for hyperparameter tuning
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=5, n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train, y_train)

# Get the best parameters and model for Random Forest
best_rf = grid_search_rf.best_estimator_
best_rf_params = grid_search_rf.best_params_
best_rf_accuracy = grid_search_rf.best_score_

# Save the best Random Forest model to a file
rf_model_filename = 'best_random_forest_model.pkl'
with open(rf_model_filename, 'wb') as file:
    pickle.dump(best_rf, file)

# Get the best parameters and model for XGBoost
best_xgb = grid_search_xgb.best_estimator_
best_xgb_params = grid_search_xgb.best_params_
best_xgb_accuracy = grid_search_xgb.best_score_

# Save the best XGBoost model to a file
xgb_model_filename = 'best_xgboost_model.pkl'
with open(xgb_model_filename, 'wb') as file:
    pickle.dump(best_xgb, file)

# Save all accuracies to a text file
with open('accuracies_for_each_case.txt', 'w') as f:
    f.write(f"Random Forest Best Parameters: {best_rf_params}\n")
    f.write(f"Random Forest Best Accuracy: {best_rf_accuracy:.4f}\n\n")
    f.write(f"XGBoost Best Parameters: {best_xgb_params}\n")
    f.write(f"XGBoost Best Accuracy: {best_xgb_accuracy:.4f}\n\n")

# Load and test the best models
with open(rf_model_filename, 'rb') as file:
    loaded_rf_model = pickle.load(file)
rf_pred_loaded = loaded_rf_model.predict(X_test)
rf_accuracy_loaded = accuracy_score(y_test, rf_pred_loaded)

with open(xgb_model_filename, 'rb') as file:
    loaded_xgb_model = pickle.load(file)
xgb_pred_loaded = loaded_xgb_model.predict(X_test)
xgb_accuracy_loaded = accuracy_score(y_test, xgb_pred_loaded)

# Output the results
print(f"Loaded Random Forest Model Accuracy: {rf_accuracy_loaded:.4f}")
print(f"Loaded XGBoost Model Accuracy: {xgb_accuracy_loaded:.4f}")

# Identify and print misclassified images for Random Forest
print("Random Forest Misclassified Images:")
rf_misclassified_indices = np.where(rf_pred_loaded != y_test)[0]
for index in rf_misclassified_indices:
    print(f"File: {file_test[index]}, Predicted: {label_encoder.inverse_transform([rf_pred_loaded[index]])[0]}, Actual: {label_encoder.inverse_transform([y_test[index]])[0]}")

# Identify and print misclassified images for XGBoost
print("XGBoost Misclassified Images:")
xgb_misclassified_indices = np.where(xgb_pred_loaded != y_test)[0]
for index in xgb_misclassified_indices:
    print(f"File: {file_test[index]}, Predicted: {label_encoder.inverse_transform([xgb_pred_loaded[index]])[0]}, Actual: {label_encoder.inverse_transform([y_test[index]])[0]}")

# Return the best parameters and accuracy for Random Forest and XGBoost
results = {
    "Random Forest Best Parameters": best_rf_params,
    "Random Forest Best Accuracy": best_rf_accuracy,
    "XGBoost Best Parameters": best_xgb_params,
    "XGBoost Best Accuracy": best_xgb_accuracy
}

print(results)