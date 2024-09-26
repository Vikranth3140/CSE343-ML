import os
import pandas as pd

# Paths to the necessary files and directories
label_file_path = '/home/dhruv/Roamify/ML-2/images_dataset/label.csv'
image_directory = '/home/dhruv/Roamify/ML-2/images_dataset/data'
misclassified_file_path = '/home/dhruv/Roamify/ML-2/images_dataset/missclassified.txt'

# Load the misclassified images from the text file
misclassified_images = []

with open(misclassified_file_path, 'r') as file:
    for line in file:
        # Extract the image filename from the line
        image_name = line.split(',')[0].replace('File: ', '').strip()
        misclassified_images.append(image_name)

# Load the label file
labels_df = pd.read_csv(label_file_path)

# Filter out the misclassified images from the labels DataFrame
labels_df_filtered = labels_df[~labels_df['filename'].isin(misclassified_images)]

# Save the updated labels DataFrame back to the CSV file
labels_df_filtered.to_csv(label_file_path, index=False)

# Delete the misclassified images from the image directory
for image_name in misclassified_images:
    image_path = os.path.join(image_directory, image_name)
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Deleted: {image_path}")
    else:
        print(f"File not found: {image_path}")

print("Finished removing misclassified images from the dataset.")
