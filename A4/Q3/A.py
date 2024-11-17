import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Select 3 classes (e.g., 'cat', 'dog', 'bird')
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

# Define a custom dataset class
class CustomCIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label

# Wrap filtered datasets into the custom dataset class
custom_train_dataset = CustomCIFAR10Dataset(filtered_train_dataset)
custom_test_dataset = CustomCIFAR10Dataset(filtered_test_dataset)

# Stratified split of training dataset (80% train, 20% validation)
train_len = int(0.8 * len(custom_train_dataset))
val_len = len(custom_train_dataset) - train_len
train_dataset, val_dataset = random_split(custom_train_dataset, [train_len, val_len])

# Stratified sampling for the test dataset (ensure 1,000 samples per class)
test_indices = []
class_counts = {cls_idx: 0 for cls_idx in selected_idx}
for idx, (_, label) in enumerate(custom_test_dataset):
    if class_counts[label] < 1000:
        test_indices.append(idx)
        class_counts[label] += 1
        if all(count == 1000 for count in class_counts.values()):
            break
custom_test_dataset = Subset(custom_test_dataset, test_indices)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(custom_test_dataset, batch_size=64, shuffle=False)

# Print dataset sizes
print(f"Train Dataset: {len(train_dataset)} samples")
print(f"Validation Dataset: {len(val_dataset)} samples")
print(f"Test Dataset: {len(custom_test_dataset)} samples")