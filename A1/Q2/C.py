import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import os
import seaborn as sns

confusion_matrix_plot = 'Plots/'
os.makedirs(confusion_matrix_plot, exist_ok=True)

df = pd.read_csv('Heart Disease.csv')

# df.fillna(df.median(), inplace=True)
df.fillna(df.mean(), inplace=True)

numerical_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

X = df.drop('HeartDisease', axis=1).values
y = df['HeartDisease'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Logistic Regression using Batch Gradient Descent
def logistic_regression_extended(X, y, X_val, y_val, lr=0.00001, iterations=10000):
    # weights = np.zeros(X.shape[1])
    weights = np.random.rand(X.shape[1])
    bias = 0
    m = len(y)
    
    train_losses = []
    val_losses = []
    
    for i in range(iterations):
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)

        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        
        weights -= lr * dw
        bias -= lr * db
        
        train_loss = cross_entropy_loss(y, y_pred)
        val_pred = sigmoid(np.dot(X_val, weights) + bias)
        val_loss = cross_entropy_loss(y_val, val_pred)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    return weights, bias, train_losses, val_losses

# Min-Max Scaling function
def min_max_scale(X_train, X_val):
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    
    X_train_scaled = (X_train - X_min) / (X_max - X_min)
    X_val_scaled = (X_val - X_min) / (X_max - X_min)
    
    return X_train_scaled, X_val_scaled

def apply_scaling(scaling=True):
    if scaling:
        X_train_scaled, X_val_scaled = min_max_scale(X_train, X_val)
        return X_train_scaled, X_val_scaled
    else:
        return X_train, X_val

def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)

X_train_scaled, X_val_scaled = apply_scaling(scaling=True)
weights_minmax, bias_minmax, train_losses_minmax, val_losses_minmax = logistic_regression_extended(X_train_scaled, y_train, X_val_scaled, y_val)
X_train_unscaled, X_val_unscaled = apply_scaling(scaling=False)
weights_no_scale, bias_no_scale, train_losses_no_scaling, val_losses_no_scaling = logistic_regression_extended(X_train_unscaled, y_train, X_val_unscaled, y_val)

y_val_pred_probs = predict(X_val_scaled, weights_minmax, bias_minmax)
y_val_pred = (y_val_pred_probs >= 0.5).astype(int)

conf_matrix = confusion_matrix(y_val, y_val_pred)

precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
roc_auc = roc_auc_score(y_val, y_val_pred_probs)

print("Results for Min-Max Scaled Data:")
print("Confusion Matrix:\n", conf_matrix)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC Score: {roc_auc}")

train_accuracy_scaled = (y_train == (predict(X_train_scaled, weights_minmax, bias_minmax) >= 0.5).astype(int)).mean()
val_accuracy_scaled = (y_val == (predict(X_val_scaled, weights_minmax, bias_minmax) >= 0.5).astype(int)).mean()
print(f"Training Accuracy (Scaled): {train_accuracy_scaled}")
print(f"Validation Accuracy (Scaled): {val_accuracy_scaled}")
print("\n")

y_val_pred_probs_unscaled = predict(X_val_unscaled, weights_no_scale, bias_no_scale)
y_val_pred_unscaled = (y_val_pred_probs_unscaled >= 0.5).astype(int)

conf_matrix_unscaled = confusion_matrix(y_val, y_val_pred_unscaled)
precision_unscaled = precision_score(y_val, y_val_pred_unscaled)
recall_unscaled = recall_score(y_val, y_val_pred_unscaled)
f1_unscaled = f1_score(y_val, y_val_pred_unscaled)
roc_auc_unscaled = roc_auc_score(y_val, y_val_pred_probs_unscaled)

print("Results for Unscaled Data:")
print("Confusion Matrix (Unscaled):\n", conf_matrix_unscaled)
print(f"Precision (Unscaled): {precision_unscaled}")
print(f"Recall (Unscaled): {recall_unscaled}")
print(f"F1 Score (Unscaled): {f1_unscaled}")
print(f"ROC-AUC Score (Unscaled): {roc_auc_unscaled}")

train_accuracy_unscaled = (y_train == (predict(X_train_unscaled, weights_no_scale, bias_no_scale) >= 0.5).astype(int)).mean()
val_accuracy_unscaled = (y_val == (predict(X_val_unscaled, weights_no_scale, bias_no_scale) >= 0.5).astype(int)).mean()
print(f"Training Accuracy (Unscaled): {train_accuracy_unscaled}")
print(f"Validation Accuracy (Unscaled): {val_accuracy_unscaled}")

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix (Scaled Data)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_unscaled, annot=True, fmt="d", cmap="Reds", cbar=False)
plt.title("Confusion Matrix (Unscaled Data)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(confusion_matrix_plot, "confusion_matrix_plot.png"))
plt.show()