import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('Heart Disease.csv')

df.fillna(df.mean(), inplace=True)
X = df.drop('HeartDisease', axis=1).values
y = df['HeartDisease'].values

# Split the dataset into 70:15:15 train, test and validation splits
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def logistic_regression(X, y, X_val, y_val, lr=0.1, iterations=100):
    weights = np.zeros(X.shape[1])
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
        
    return train_losses, val_losses

def min_max_scale(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min)

X_train_scaled = min_max_scale(X_train)
X_val_scaled = min_max_scale(X_val)

train_losses_no_scaling, val_losses_no_scaling = logistic_regression(X_train, y_train, X_val, y_val)
train_losses_minmax, val_losses_minmax = logistic_regression(X_train_scaled, y_train, X_val_scaled, y_val)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses_no_scaling, label='Training Loss (No Scaling)')
plt.plot(val_losses_no_scaling, label='Validation Loss (No Scaling)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs. Iterations (No Scaling)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses_minmax, label='Training Loss (Min-Max Scaling)')
plt.plot(val_losses_minmax, label='Validation Loss (Min-Max Scaling)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs. Iterations (Min-Max Scaling)')
plt.legend()

plt.tight_layout()
plt.show()