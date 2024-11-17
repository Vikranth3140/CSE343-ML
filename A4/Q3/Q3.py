import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Select 3 classes
selected_classes = ['cat', 'dog', 'bird']
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(train_dataset.classes)}
selected_idx = [class_to_idx[cls] for cls in selected_classes]

# Function to filter datasets
def filter_dataset(dataset, selected_idx):
    indices = [i for i, (_, label) in enumerate(dataset) if label in selected_idx]
    return Subset(dataset, indices)

# Filter train and test datasets
filtered_train_dataset = filter_dataset(train_dataset, selected_idx)
filtered_test_dataset = filter_dataset(test_dataset, selected_idx)

# Check sizes
print(f"Filtered Train Dataset: {len(filtered_train_dataset)} samples")
print(f"Filtered Test Dataset: {len(filtered_test_dataset)} samples")

# Split train dataset to get exactly 15,000 samples
train_indices = np.random.choice(len(filtered_train_dataset), 15000, replace=False)
filtered_train_dataset = Subset(filtered_train_dataset, train_indices)

# Split test dataset to get exactly 3,000 samples
test_indices = np.random.choice(len(filtered_test_dataset), 3000, replace=False)
filtered_test_dataset = Subset(filtered_test_dataset, test_indices)

# Create DataLoaders
train_loader = DataLoader(filtered_train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(filtered_test_dataset, batch_size=64, shuffle=False)

# Verify dataset distributions
print(f"Final Train Dataset: {len(filtered_train_dataset)} samples")
print(f"Final Test Dataset: {len(filtered_test_dataset)} samples")