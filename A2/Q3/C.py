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

os.makedirs('Plots', exist_ok=True)

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

hog_features_list = []

    
for index, row in labels_df.iterrows():
    image_path = os.path.join(image_directory, row['filename'])
    
    # Extract HOG features
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

# Initialize the models
models = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Perceptron": Perceptron(random_state=42)
}

# Dictionary to store accuracy results
results = {}

# Train and test each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Test the model
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store the results
    results[model_name] = accuracy

# Display the results
for model_name, accuracy in results.items():
    print(f"Model: {model_name}, Accuracy: {accuracy:.4f}")