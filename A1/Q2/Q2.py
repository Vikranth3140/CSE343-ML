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







import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

plots = 'Plots/'
os.makedirs(plots, exist_ok=True)

df = pd.read_csv('Heart Disease.csv')

df.fillna(df.median(), inplace=True)

numerical_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

X = df.drop('HeartDisease', axis=1).values
y = df['HeartDisease'].values

# Split the dataset into 70:15:15 train, test, and validation splits
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Logistic Regression using Batch Gradient Descent
def logistic_regression(X, y, X_val, y_val, lr=0.00001, iterations=10000):
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

X_train_scaled, X_val_scaled = apply_scaling(scaling=True)
train_losses_minmax, val_losses_minmax = logistic_regression(X_train_scaled, y_train, X_val_scaled, y_val)

X_train_unscaled, X_val_unscaled = apply_scaling(scaling=False)
train_losses_no_scaling, val_losses_no_scaling = logistic_regression(X_train_unscaled, y_train, X_val_unscaled, y_val)

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
plt.savefig(os.path.join(plots, "loss_iterations_plots.png"))
plt.show()




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import os
import seaborn as sns

plots = 'Plots/'
os.makedirs(plots, exist_ok=True)

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
def logistic_regression_extended(X, y, X_val, y_val, lr=0.0001, iterations=10000):
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
weights_minmax, bias_minmax, train_losses_minmax, val_losses_minmax = logistic_regression_extended(X_train_scaled, y_train, X_val_scaled, y_val, lr=0.0001, iterations=10000)
X_train_unscaled, X_val_unscaled = apply_scaling(scaling=False)
weights_no_scale, bias_no_scale, train_losses_no_scaling, val_losses_no_scaling = logistic_regression_extended(X_train_unscaled, y_train, X_val_unscaled, y_val, lr=0.0001, iterations=1000)

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

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
axes[0].set_title("Confusion Matrix (Scaled Data)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(conf_matrix_unscaled, annot=True, fmt="d", cmap="Reds", cbar=False, ax=axes[1])
axes[1].set_title("Confusion Matrix (Unscaled Data)")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.savefig(os.path.join(plots, "confusion_matrices_plot.png"))
plt.tight_layout()
plt.show()




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

    weights, bias, a, b, c, d = mini_batch_gradient_descent(X_train_fold, y_train_fold, X_val_fold, y_val_fold, lr=0.0001, iterations=100, batch_size=8)
    # weights, bias, a, b, c, d = stochastic_gradient_descent(X_train_fold, y_train_fold, X_val_fold, y_val_fold, lr=0.0001, iterations=100)

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

folds = range(1, k+1)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(folds, accuracy_scores, marker='o', linestyle='--')
plt.title('Accuracy Across Folds')
plt.xlabel('Fold')
plt.ylabel('Accuracy')

plt.subplot(2, 2, 2)
plt.plot(folds, precision_scores, marker='o', linestyle='--')
plt.title('Precision Across Folds')
plt.xlabel('Fold')
plt.ylabel('Precision')

plt.subplot(2, 2, 3)
plt.plot(folds, recall_scores, marker='o', linestyle='--')
plt.title('Recall Across Folds')
plt.xlabel('Fold')
plt.ylabel('Recall')

plt.subplot(2, 2, 4)
plt.plot(folds, f1_scores, marker='o', linestyle='--')
plt.title('F1 Score Across Folds')
plt.xlabel('Fold')
plt.ylabel('F1 Score')

plt.savefig(os.path.join(plots, "metrics_across_folds.png"))

plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from tqdm import tqdm

plots = 'Plots/'
experiments_dir = os.path.join(plots, 'Experiments/')
os.makedirs(plots, exist_ok=True)
os.makedirs(experiments_dir, exist_ok=True)

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

# weights_es, bias_es, train_losses_es, val_losses_es, train_acc_es, val_acc_es = mini_batch_gradient_descent_early_stopping(X_train, y_train, X_val, y_val, lr=0.0001, iterations=500, batch_size=256, l1_ratio=0.0, l2_ratio=0.0, early_stopping=True, patience=20)
# weights_no_es, bias_no_es, train_losses_no_es, val_losses_no_es, train_acc_no_es, val_acc_no_es = mini_batch_gradient_descent_early_stopping(X_train, y_train, X_val, y_val, lr=0.0001, iterations=500, batch_size=256, l1_ratio=0.0, l2_ratio=0.0, early_stopping=False)

def plot_figure(train_metric, val_metric, metric_name, label, filename, save_dir):
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

    plt.savefig(os.path.join(save_dir, filename))

weights_es, bias_es, train_losses_es, val_losses_es, train_acc_es, val_acc_es = mini_batch_gradient_descent_early_stopping(
    X_train, y_train, X_val, y_val, lr=0.0001, iterations=500, batch_size=256, l1_ratio=0.0, l2_ratio=0.0, early_stopping=True, patience=20)

weights_no_es, bias_no_es, train_losses_no_es, val_losses_no_es, train_acc_no_es, val_acc_no_es = mini_batch_gradient_descent_early_stopping(
    X_train, y_train, X_val, y_val, lr=0.0001, iterations=500, batch_size=256, l1_ratio=0.0, l2_ratio=0.0, early_stopping=False)

plot_figure(train_losses_es, val_losses_es, 'Loss', 'Early Stopping', 'loss_early_stopping.png', plots)
plot_figure(train_acc_es, val_acc_es, 'Accuracy', 'Early Stopping', 'accuracy_early_stopping.png', plots)

plot_figure(train_losses_no_es, val_losses_no_es, 'Loss', 'No Early Stopping', 'loss_no_early_stopping.png', plots)
plot_figure(train_acc_no_es, val_acc_no_es, 'Accuracy', 'No Early Stopping', 'accuracy_no_early_stopping.png', plots)

learning_rates = [0.0001, 0.001]
l1_ratios = [0.0, 0.1]
l2_ratios = [0.0, 0.1]

results = []

for lr in learning_rates:
    for l1 in l1_ratios:
        for l2 in l2_ratios:
            weights, bias, train_losses, val_losses, train_acc, val_acc = mini_batch_gradient_descent_early_stopping(
                X_train, y_train, X_val, y_val, lr=lr, iterations=500, batch_size=32, l1_ratio=l1, l2_ratio=l2, early_stopping=True, patience=20)
            
            label = f'LR={lr}, L1={l1}, L2={l2}'
            plot_figure(train_losses, val_losses, 'Loss', label, f'loss_{lr}_{l1}_{l2}.png', experiments_dir)
            plot_figure(train_acc, val_acc, 'Accuracy', label, f'accuracy_{lr}_{l1}_{l2}.png', experiments_dir)