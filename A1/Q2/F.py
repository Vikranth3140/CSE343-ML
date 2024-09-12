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

# L1 and L2 regularization in the loss function
def regularized_loss(y, y_pred, weights, l1_ratio=0.0, l2_ratio=0.0):
    cross_entropy = cross_entropy_loss(y, y_pred)
    l1_penalty = l1_ratio * np.sum(np.abs(weights))
    l2_penalty = l2_ratio * np.sum(np.square(weights))
    return cross_entropy + l1_penalty + l2_penalty

# Early Stopping
def check_early_stopping(val_loss, best_loss, patience_counter, patience, delta=0):
    if best_loss is None or val_loss < best_loss - delta:
        return val_loss, 0, False
    else:
        patience_counter += 1
        if patience_counter >= patience:
            return best_loss, patience_counter, True
        return best_loss, patience_counter, False

# Mini-Batch Gradient Descent with L1 and L2 Regularization with Early Stopping
def mini_batch_gradient_descent_early_stopping(X, y, X_val, y_val, lr=0.01, iterations=500, batch_size=32, l1_ratio=0.0, l2_ratio=0.0, early_stopping=True, patience=10):
    weights = np.random.rand(X.shape[1])
    bias = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    m = len(y)
    num_batches = m // batch_size
    best_loss = None
    patience_counter = 0
    
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
            
            weights -= lr * (dw.flatten() + l1_ratio * np.sign(weights) + l2_ratio * 2 * weights)
            bias -= lr * db
        
        train_loss = regularized_loss(y, sigmoid(np.dot(X, weights) + bias), weights, l1_ratio, l2_ratio)
        val_loss = regularized_loss(y_val, sigmoid(np.dot(X_val, weights) + bias), weights, l1_ratio, l2_ratio)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        train_accuracy = calculate_accuracy(X, y, weights, bias)
        val_accuracy = calculate_accuracy(X_val, y_val, weights, bias)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        if early_stopping:
            best_loss, patience_counter, stop = check_early_stopping(val_loss, best_loss, patience_counter, patience)
            if stop:
                print("Early stopping triggered.")
                break
    
    return weights, bias, train_losses, val_losses, train_accuracies, val_accuracies

weights_es, bias_es, train_losses_es, val_losses_es, train_acc_es, val_acc_es = mini_batch_gradient_descent_early_stopping(X_train, y_train, X_val, y_val, lr=0.0001, iterations=500, batch_size=256, l1_ratio=0.0, l2_ratio=0.0, early_stopping=True, patience=20)
weights_no_es, bias_no_es, train_losses_no_es, val_losses_no_es, train_acc_no_es, val_acc_no_es = mini_batch_gradient_descent_early_stopping(X_train, y_train, X_val, y_val, lr=0.0001, iterations=500, batch_size=256, l1_ratio=0.0, l2_ratio=0.0, early_stopping=False)

def plot_figure(train_metric, val_metric, metric_name, label, filename):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_metric, label=f'Training {metric_name} ({label})', color='blue')
    plt.xlabel('Iterations')
    plt.ylabel(metric_name)
    plt.title(f'Training {metric_name} vs. Iterations ({label})')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_metric, label=f'Validation {metric_name} ({label})', color='red')
    plt.xlabel('Iterations')
    plt.ylabel(metric_name)
    plt.title(f'Validation {metric_name} vs. Iterations ({label})')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(plots, filename))

    plt.tight_layout()
    plt.show()

plot_figure(train_losses_no_es, val_losses_no_es, 'Loss', 'No Early Stopping', 'loss_no_early_stopping.png')
plot_figure(train_acc_no_es, val_acc_no_es, 'Accuracy', 'No Early Stopping', 'accuracy_no_early_stopping.png')

plot_figure(train_losses_es, val_losses_es, 'Loss', 'Early Stopping', 'loss_early_stopping.png')
plot_figure(train_acc_es, val_acc_es, 'Accuracy', 'Early Stopping', 'accuracy_early_stopping.png')