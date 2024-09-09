import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset
df = pd.read_csv('Heart Disease.csv')

# Fill missing values with median
df.fillna(df.median(), inplace=True)

# Select numerical columns
numerical_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

# Prepare the feature matrix X and target vector y
X = df.drop('HeartDisease', axis=1)[numerical_cols].values
y = df['HeartDisease'].values

# Split the dataset into train (70%), test (15%), and validation (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cross-entropy loss
def cross_entropy_loss(y, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Clipping to avoid log(0)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Logistic Regression using Batch Gradient Descent
def logistic_regression(X, y, X_val, y_val, lr=0.01, iterations=1000):
    weights = np.zeros(X.shape[1])
    bias = 0
    m = len(y)
    
    train_losses = []
    val_losses = []
    
    for i in range(iterations):
        # Compute predictions
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
        
        # Compute gradients
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        
        # Update weights and bias
        weights -= lr * dw
        bias -= lr * db
        
        # Compute training and validation loss
        train_loss = cross_entropy_loss(y, y_pred)
        val_pred = sigmoid(np.dot(X_val, weights) + bias)
        val_loss = cross_entropy_loss(y_val, val_pred)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if i % 100 == 0:
            print(f"Iteration {i}: Training Loss = {train_loss}, Validation Loss = {val_loss}")
    
    return weights, bias, train_losses, val_losses

# Min-max scaling
def min_max_scale(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min)

# Scale the features
X_train_scaled = min_max_scale(X_train)
X_val_scaled = min_max_scale(X_val)

# Train the model without scaling and with min-max scaling
weights_no_scaling, bias_no_scaling, train_losses_no_scaling, val_losses_no_scaling = logistic_regression(X_train, y_train, X_val, y_val)
weights_minmax, bias_minmax, train_losses_minmax, val_losses_minmax = logistic_regression(X_train_scaled, y_train, X_val_scaled, y_val)

# Predicting on validation set for Min-Max scaling
y_val_pred_prob = sigmoid(np.dot(X_val_scaled, weights_minmax) + bias_minmax)
y_val_pred = (y_val_pred_prob >= 0.5).astype(int)

# Confusion matrix
conf_matrix = confusion_matrix(y_val, y_val_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

# Extract TP, TN, FP, FN
tn, fp, fn, tp = conf_matrix.ravel()

# Precision, Recall, F1 Score, ROC-AUC
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
roc_auc = roc_auc_score(y_val, y_val_pred_prob)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")