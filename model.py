import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load the features from CSV
features_df = pd.read_csv('extracted_features.csv')

# Separate features, labels, and filenames
X = features_df.drop(columns=['label', 'filename']).values
y = LabelEncoder().fit_transform(features_df['label'].values)
file_names = features_df['filename'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
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
    print(f"File: {file_names[index]}, Predicted: {LabelEncoder().inverse_transform([y_pred[index]])[0]}, Actual: {LabelEncoder().inverse_transform([y_test[index]])[0]}")