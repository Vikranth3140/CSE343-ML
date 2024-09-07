import pandas as pd
from sklearn.preprocessing import LabelEncoder
from umap import UMAP
import matplotlib.pyplot as plt
import os

umap_dir = 'UMAP/'
os.makedirs(umap_dir, exist_ok=True)

data = pd.read_csv('Electricity Bill.csv')

label_encoder = LabelEncoder()
categorical_columns = ['Building_Type', 'Building_Status', 'Maintenance_Priority', 'Green_Certified']
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

features = data.drop(columns=['Electricity_Bill'])

# Apply UMAP to reduce to 2 dimensions
umap_model = UMAP(n_components=2, random_state=42)
reduced_data = umap_model.fit_transform(features)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data['Electricity_Bill'], cmap='viridis', s=50)
plt.colorbar(label='Electricity Bill')
plt.title('UMAP Dimensionality Reduction to 2D')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig(os.path.join(umap_dir, "umap_projection.png"))
plt.show()