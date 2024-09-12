import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from tqdm import tqdm

plots = 'Plots/'
os.makedirs(plots, exist_ok=True)

df = pd.read_csv('Heart Disease.csv')

df.fillna(df.mean(), inplace=True)

X = df.drop('HeartDisease', axis=1).values
y = df['HeartDisease'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def calculate_accuracy(X, y, weights, bias):
    y_pred = sigmoid(np.dot(X, weights) + bias)
    y_pred_class = (y_pred >= 0.5).astype(int)
    return accuracy_score(y, y_pred_class)

# Stochastic Gradient Descent (SGD)
def stochastic_gradient_descent(X, y, X_val, y_val, lr=0.001, iterations=100):
    weights = np.random.rand(X.shape[1])
    bias = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for i in tqdm(range(iterations), desc="SGD Iterations"):
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        
        for idx in indices:
            xi = X[idx].reshape(1, -1)
            yi = y[idx]
            
            z = np.dot(xi, weights) + bias
            y_pred = sigmoid(z)
            
            dw = np.dot(xi.T, (y_pred - yi))
            db = y_pred - yi
            
            weights -= lr * dw.flatten()
            bias -= lr * db
        
        train_loss = cross_entropy_loss(y, sigmoid(np.dot(X, weights) + bias))
        val_loss = cross_entropy_loss(y_val, sigmoid(np.dot(X_val, weights) + bias))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        train_accuracy = calculate_accuracy(X, y, weights, bias)
        val_accuracy = calculate_accuracy(X_val, y_val, weights, bias)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
    
    return weights, bias, train_losses, val_losses, train_accuracies, val_accuracies

# Mini-Batch Gradient Descent (MBGD)
def mini_batch_gradient_descent(X, y, X_val, y_val, lr=0.01, iterations=500, batch_size=32):
    weights = np.random.rand(X.shape[1])
    bias = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    m = len(y)
    num_batches = m // batch_size
    
    for i in tqdm(range(iterations), desc=f"MBGD (Batch Size {batch_size}) Iterations"):
        indices = np.arange(m)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        for j in range(num_batches):
            start = j * batch_size
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]
            
            z = np.dot(X_batch, weights) + bias
            y_pred = sigmoid(z)
            
            dw = np.dot(X_batch.T, (y_pred - y_batch)) / batch_size
            db = np.sum(y_pred - y_batch) / batch_size
            
            weights -= lr * dw
            bias -= lr * db
        
        train_loss = cross_entropy_loss(y, sigmoid(np.dot(X, weights) + bias))
        val_loss = cross_entropy_loss(y_val, sigmoid(np.dot(X_val, weights) + bias))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        train_accuracy = calculate_accuracy(X, y, weights, bias)
        val_accuracy = calculate_accuracy(X_val, y_val, weights, bias)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
    
    return weights, bias, train_losses, val_losses, train_accuracies, val_accuracies

# k-fold cross-validation
k = 5
fold_size = len(X) // k

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for i in range(k):
    X_val_fold = X[i * fold_size:(i + 1) * fold_size]
    y_val_fold = y[i * fold_size:(i + 1) * fold_size]

    X_train_fold = np.concatenate((X[:i * fold_size], X[(i + 1) * fold_size:]), axis=0)
    y_train_fold = np.concatenate((y[:i * fold_size], y[(i + 1) * fold_size:]), axis=0)

    # weights, bias, a, b, c, d = mini_batch_gradient_descent(X_train_fold, y_train_fold, X_val_fold, y_val_fold, lr=0.0001, iterations=100, batch_size=32)
    weights, bias, a, b, c, d = stochastic_gradient_descent(X_train_fold, y_train_fold, X_val_fold, y_val_fold, lr=0.0001, iterations=100)

    y_val_pred_probs = sigmoid(np.dot(X_val_fold, weights) + bias)
    y_val_pred = (y_val_pred_probs >= 0.5).astype(int)

    accuracy_scores.append(accuracy_score(y_val_fold, y_val_pred))
    precision_scores.append(precision_score(y_val_fold, y_val_pred, zero_division=1))
    recall_scores.append(recall_score(y_val_fold, y_val_pred))
    f1_scores.append(f1_score(y_val_fold, y_val_pred, zero_division=1))

accuracy_scores = np.array(accuracy_scores)
precision_scores = np.array(precision_scores)
recall_scores = np.array(recall_scores)
f1_scores = np.array(f1_scores)

avg_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
avg_precision = np.mean(precision_scores)
std_precision = np.std(precision_scores)
avg_recall = np.mean(recall_scores)
std_recall = np.std(recall_scores)
avg_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print(f"Accuracy: Mean = {avg_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Precision: Mean = {avg_precision:.4f} ± {std_precision:.4f}")
print(f"Recall: Mean = {avg_recall:.4f} ± {std_recall:.4f}")
print(f"F1 Score: Mean = {avg_f1:.4f} ± {std_f1:.4f}")