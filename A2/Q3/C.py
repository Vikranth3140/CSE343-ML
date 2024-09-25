import cv2
import pandas as pd
import os
from skimage.feature import hog
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

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

# Load images and extract features
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

# Apply PCA to reduce dimensionality
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X)

# Split the dataset into an 80:20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Initialize base models for stacking
base_models = [
    ('naive_bayes', GaussianNB()),
    ('decision_tree', DecisionTreeClassifier(random_state=42)),
    ('random_forest', RandomForestClassifier(random_state=42)),
    ('perceptron', Perceptron(random_state=42))
]

# Create the Stacking Classifier
stacked_model = StackingClassifier(estimators=base_models, final_estimator=RandomForestClassifier(random_state=42))

# Train the stacked model
stacked_model.fit(X_train, y_train)

# Test the model and calculate accuracy
y_pred = stacked_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Stacked Model Accuracy after PCA: {accuracy:.4f}")