import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

weights_sgd, bias_sgd, train_losses_sgd, val_losses_sgd, train_acc_sgd, val_acc_sgd = stochastic_gradient_descent(X_train, y_train, X_val, y_val, lr=0.0001, iterations=100)
weights_mbgd_32, bias_mbgd_32, train_losses_mbgd_8, val_losses_mbgd_8, train_acc_mbgd_8, val_acc_mbgd_8 = mini_batch_gradient_descent(X_train, y_train, X_val, y_val, lr=0.0001, iterations=100, batch_size=8)
weights_mbgd_32, bias_mbgd_32, train_losses_mbgd_32, val_losses_mbgd_32, train_acc_mbgd_32, val_acc_mbgd_32 = mini_batch_gradient_descent(X_train, y_train, X_val, y_val, lr=0.0001, iterations=100, batch_size=32)
weights_mbgd_64, bias_mbgd_64, train_losses_mbgd_64, val_losses_mbgd_64, train_acc_mbgd_64, val_acc_mbgd_64 = mini_batch_gradient_descent(X_train, y_train, X_val, y_val, lr=0.0001, iterations=100, batch_size=64)
weights_mbgd_256, bias_mbgd_256, train_losses_mbgd_256, val_losses_mbgd_256, train_acc_mbgd_256, val_acc_mbgd_256 = mini_batch_gradient_descent(X_train, y_train, X_val, y_val, lr=0.0001, iterations=100, batch_size=256)
weights_mbgd_512, bias_mbgd_512, train_losses_mbgd_512, val_losses_mbgd_512, train_acc_mbgd_512, val_acc_mbgd_512 = mini_batch_gradient_descent(X_train, y_train, X_val, y_val, lr=0.0001, iterations=100, batch_size=512)

def plot_figure(train_metric, val_metric, metric_name, batch_size_label, filename):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_metric, label=f'Training {metric_name}', color='blue')
    plt.xlabel('Iterations')
    plt.ylabel(metric_name)
    plt.title(f'Training {metric_name} vs. Iterations ({batch_size_label})')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_metric, label=f'Validation {metric_name}', color='red')
    plt.xlabel('Iterations')
    plt.ylabel(metric_name)
    plt.title(f'Validation {metric_name} vs. Iterations ({batch_size_label})')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(plots, filename))

    plt.tight_layout()
    plt.show()

plot_figure(train_losses_sgd, val_losses_sgd, 'Loss', 'SGD', 'sgd_loss.png')
plot_figure(train_acc_sgd, val_acc_sgd, 'Accuracy', 'SGD', 'sgd_accuracy.png')

plot_figure(train_losses_mbgd_8, val_losses_mbgd_8, 'Loss', 'MBGD (Batch Size 8)', 'mbgd_8_loss.png')
plot_figure(train_acc_mbgd_8, val_acc_mbgd_8, 'Accuracy', 'MBGD (Batch Size 8)', 'mbgd_8_accuracy.png')

plot_figure(train_losses_mbgd_32, val_losses_mbgd_32, 'Loss', 'MBGD (Batch Size 32)', 'mbgd_32_loss.png')
plot_figure(train_acc_mbgd_32, val_acc_mbgd_32, 'Accuracy', 'MBGD (Batch Size 32)', 'mbgd_32_accuracy.png')

plot_figure(train_losses_mbgd_64, val_losses_mbgd_64, 'Loss', 'MBGD (Batch Size 64)', 'mbgd_64_loss.png')
plot_figure(train_acc_mbgd_64, val_acc_mbgd_64, 'Accuracy', 'MBGD (Batch Size 64)', 'mbgd_64_accuracy.png')

plot_figure(train_losses_mbgd_256, val_losses_mbgd_256, 'Loss', 'MBGD (Batch Size 256)', 'mbgd_256_loss.png')
plot_figure(train_acc_mbgd_256, val_acc_mbgd_256, 'Accuracy', 'MBGD (Batch Size 256)', 'mbgd_256_accuracy.png')

plot_figure(train_losses_mbgd_512, val_losses_mbgd_512, 'Loss', 'MBGD (Batch Size 512)', 'mbgd_512_loss.png')
plot_figure(train_acc_mbgd_512, val_acc_mbgd_512, 'Accuracy', 'MBGD (Batch Size 512)', 'mbgd_512_accuracy.png')