import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

import seaborn as sns
import random
from PIL import Image
import matplotlib.pyplot as plt





# a

labels_df = pd.read_csv('label.csv')

# Dataset Overview
print(f"Total number of images: {len(labels_df)}")
print("Number of images per class:\n", labels_df['label'].value_counts())

# Check for missing values
print("\nMissing values:\n", labels_df.isnull().sum())

image_directory = 'data'

image_sizes = []
for filename in labels_df['filename']:
    image_path = os.path.join(image_directory, filename)
    with Image.open(image_path) as img:
        image_sizes.append(img.size)

image_size_df = pd.DataFrame(image_sizes, columns=['width', 'height'])
print("\nImage Size Statistics:")
print(image_size_df.describe())















# b

# Plotting the class distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=labels_df, x='label', order=labels_df['label'].value_counts().index)
plt.xticks(rotation=90)
plt.title("Class Distribution of Human Activity Labels")
plt.show()

# Displaying a few sample images from each class
def display_sample_images(labels_df, image_directory, num_classes=5):
    sample_df = labels_df.groupby('label').apply(lambda x: x.sample(1)).reset_index(drop=True)
    
    fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))
    
    for i, (index, row) in enumerate(sample_df.iterrows()):
        if i >= num_classes:
            break
        image_path = os.path.join(image_directory, row['filename'])
        img = Image.open(image_path)
        axes[i].imshow(img)
        axes[i].set_title(row['label'])
        axes[i].axis('off')
    
    plt.show()

# Display sample images from a few classes
display_sample_images(labels_df, image_directory, num_classes=5)







# c

# Investigate class imbalance
class_distribution = labels_df['label'].value_counts(normalize=True)

# Plot class distribution to check for imbalance
plt.figure(figsize=(12, 6))
sns.barplot(x=class_distribution.index, y=class_distribution.values)
plt.xticks(rotation=90)
plt.title("Class Proportion Distribution")
plt.show()

# Threshold for imbalance (example: if a class has less than 5% of the total images)
imbalance_threshold = 0.05
imbalanced_classes = class_distribution[class_distribution < imbalance_threshold].index
print(f"Imbalanced Classes (less than {imbalance_threshold*100}% of the dataset):\n", imbalanced_classes)
