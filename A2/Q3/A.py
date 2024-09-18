import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('EDA', exist_ok=True)

# a

labels_df = pd.read_csv('label.csv')

# Dataset Overview
with open('Image_Statistics.txt', 'w') as f:
    # Overview of the dataset
    f.write(f"Total number of images: {len(labels_df)}\n\n")
    f.write("Number of images per class:\n")
    f.write(str(labels_df['label'].value_counts()) + "\n\n")
    
    f.write("\nMissing values:\n")
    f.write(str(labels_df.isnull().sum()) + "\n\n")
    
    image_directory = 'data'
    
    image_sizes = []
    for filename in labels_df['filename']:
        image_path = os.path.join(image_directory, filename)
        try:
            with Image.open(image_path) as img:
                image_sizes.append(img.size)
        except FileNotFoundError:
            f.write(f"Image not found: {filename}\n")
    
    if image_sizes:
        image_size_df = pd.DataFrame(image_sizes, columns=['width', 'height'])
        f.write("\nImage Size Statistics:\n")
        f.write(str(image_size_df.describe()) + "\n")


# b

# Plot the class distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=labels_df, x='label', order=labels_df['label'].value_counts().index)
plt.xticks(rotation=90)
plt.title("Class Distribution of Human Activity Labels")
plt.savefig('EDA/class_distribution.png')
plt.close()

# Display a few sample images from each class
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
    
    plt.savefig('EDA/sample_images.png')
    plt.close()

# Display sample images from a few classes
display_sample_images(labels_df, image_directory, num_classes=5)


# c

# Investigate class imbalance
class_distribution = labels_df['label'].value_counts(normalize=True)

plt.figure(figsize=(12, 6))
sns.barplot(x=class_distribution.index, y=class_distribution.values)
plt.xticks(rotation=90)
plt.title("Class Proportion Distribution")
plt.savefig('EDA/class_proportion_distribution.png')
plt.close()