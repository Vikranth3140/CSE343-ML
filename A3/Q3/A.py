import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV files
train_data = pd.read_csv("dataset/entire/fashion-mnist_train.csv")
test_data = pd.read_csv("dataset/entire/fashion-mnist_test.csv")

# Take the first 8000 images from the training data
train_split = train_data.iloc[:8000]

# Take the first 2000 images from the testing data
test_split = test_data.iloc[:2000]

# Optionally, save these splits to new CSV files if needed
train_split.to_csv("dataset/new/fashion-mnist_train_split.csv", index=False)
test_split.to_csv("dataset/new/fashion-mnist_test_split.csv", index=False)

# Display the shapes of the splits to confirm
print("Training Split Shape:", train_split.shape)
print("Testing Split Shape:", test_split.shape)


# Separate features and labels
train_labels = train_split['label']
train_images = train_split.drop('label', axis=1)

test_labels = test_split['label']
test_images = test_split.drop('label', axis=1)

# Normalize the data (scale pixel values to the range [0, 1])
train_images = train_images / 255.0
test_images = test_images / 255.0

# Visualize 10 random samples from the test dataset
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("10 Random Samples from Test Dataset")

for i, ax in enumerate(axes.flatten()):
    # Reshape the flat array to 28x28 for visualization
    img = test_images.iloc[i].values.reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Label: {test_labels.iloc[i]}")
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()
