import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

plots = 'Plots/'
os.makedirs(plots, exist_ok=True)

df = pd.read_csv('Heart Disease.csv')

df.fillna(df.mean(), inplace=True)

numerical_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

X = df.drop('HeartDisease', axis=1).values
y = df['HeartDisease'].values

# Split the dataset into 70:15:15 train, test and validation splits
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cross-entropy loss function
def cross_entropy_loss(y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Logistic regression using batch gradient descent
def logistic_regression(X, y, X_val, y_val, lr=0.1, iterations=100):
    weights = np.zeros(X.shape[1])
    bias = 0
    m = len(y)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for i in range(iterations):
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
        
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        
        weights -= lr * dw
        bias -= lr * db
        
        train_loss = cross_entropy_loss(y, y_pred)
        train_losses.append(train_loss)
        
        train_accuracy = np.mean((y_pred >= 0.5) == y)
        train_accuracies.append(train_accuracy)
        
        val_pred = sigmoid(np.dot(X_val, weights) + bias)
        val_loss = cross_entropy_loss(y_val, val_pred)
        val_losses.append(val_loss)
        
        val_accuracy = np.mean((val_pred >= 0.5) == y_val)
        val_accuracies.append(val_accuracy)
        
        print(f"Iteration {i+1}/{iterations}")
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        print("-" * 50)
    
    return weights, bias, train_losses, val_losses, train_accuracies, val_accuracies

weights, bias, train_losses, val_losses, train_accuracies, val_accuracies = logistic_regression(X_train, y_train, X_val, y_val)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Iterations')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy vs. Iterations')
plt.legend()

plt.savefig(os.path.join(plots, "loss_validation_plots.png"))
plt.show()