import cv2
import pandas as pd
import os
from skimage.feature import hog
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load labels
labels_df = pd.read_csv('label.csv')

# Directory containing images
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

# Prepare dataset
hog_features_list = []

for index, row in labels_df.iterrows():
    image_path = os.path.join(image_directory, row['filename'])
    hog_features = extract_hog_features(image_path)
    if hog_features is not None:
        hog_features_list.append(hog_features)

# Encode the labels into numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels_df['label'])

# Convert the HOG features list to a NumPy array
X = np.array(hog_features_list)

# Split the dataset into an 80:20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grids
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# GridSearchCV for each model
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the grid search models
rf_grid_search.fit(X_train, y_train)

# Get the best model
best_rf = rf_grid_search.best_estimator_  # Corrected to best_estimator_

print("Best parameters for Random Forest:", rf_grid_search.best_params_)

# Stacking classifier with the best estimators
stacked_model = StackingClassifier(
    estimators=[
        ('rf', best_rf)
    ],
    final_estimator=RandomForestClassifier(random_state=42)
)

# Train the stacked model
stacked_model.fit(X_train, y_train)

# Evaluate the model
y_pred = stacked_model.predict(X_test)
stacked_accuracy = accuracy_score(y_test, y_pred)
print(f"Stacked Model Accuracy after Hyperparameter Tuning: {stacked_accuracy:.4f}")