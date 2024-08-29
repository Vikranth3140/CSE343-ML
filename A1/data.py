import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import umap
import os

eda_dir = 'EDA/'
umap_dir = 'UMAP/'
os.makedirs(eda_dir, exist_ok=True)
os.makedirs(umap_dir, exist_ok=True)

df = pd.read_csv('Electricity Bill.csv')
df.columns = df.columns.str.strip()

# Split the dataset into 80:20 train and test splits
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Training set size: {train_df.shape}")
print(f"Testing set size: {test_df.shape}")

# EDA

# Pair Plots
def create_pair_plots(df, numerical_features, categorical_feature):
    plot = sns.pairplot(df[numerical_features + [categorical_feature]], hue=categorical_feature)
    plt.suptitle(f"Pair Plot - Numerical Features with {categorical_feature} as Hue", y=1.02)
    plt.savefig(os.path.join(eda_dir, f"pair_plot_{categorical_feature}.png"))
    plt.show()

# Box Plots
def create_box_plots(df, categorical_features, numerical_feature):
    plt.figure(figsize=(20, 15))
    
    for i, category in enumerate(categorical_features, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x=category, y=numerical_feature, data=df)
        plt.title(f"Box Plot - {numerical_feature} by {category}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, f"box_plot_{numerical_feature}.png"))
    plt.show()

# Violin Plots
def create_violin_plots(df, categorical_features, numerical_feature):
    plt.figure(figsize=(20, 15))
    
    for i, category in enumerate(categorical_features, 1):
        plt.subplot(2, 2, i)
        sns.violinplot(x=category, y=numerical_feature, data=df)
        plt.title(f"Violin Plot - {numerical_feature} by {category}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, f"violin_plot_{numerical_feature}.png"))
    plt.show()

# Count Plots
def create_count_plots(df, features):
    plt.figure(figsize=(20, 15))
    
    for i, feature in enumerate(features, 1):
        plt.subplot(2, 2, i)
        sns.countplot(x=feature, data=df)
        plt.title(f"Count Plot - {feature}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, f"count_plot_{feature}.png"))
    plt.show()

# Correlation Heatmap
def create_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include='number')
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(eda_dir, "correlation_heatmap.png"))
    plt.show()

categorical_features = ['Building_Type', 'Green_Certified', 'Building_Status', 'Maintenance_Priority']

numerical_features = ['Electricity_Bill', 'Energy_Consumption_Per_SqM', 'Number_of_Floors', 'Number_of_Residents']

for category in categorical_features:
    create_pair_plots(df, numerical_features, category)

create_box_plots(df, categorical_features, 'Electricity_Bill')

create_violin_plots(df, categorical_features, 'Electricity_Bill')

create_count_plots(df, categorical_features)

create_correlation_heatmap(df)



numerical_features = ['Electricity_Bill', 'Energy_Consumption_Per_SqM', 'Number_of_Floors', 'Number_of_Residents',
                      'Water_Usage_Per_Building', 'Waste_Recycled_Percentage', 'Occupancy_Rate', 'Indoor_Air_Quality',
                      'Smart_Devices_Count', 'Maintenance_Resolution_Time', 'Energy_Per_SqM']

# Apply UMAP to reduce to 2 dimensions
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_results = umap_model.fit_transform(df[numerical_features])

umap_df = pd.DataFrame(umap_results, columns=['UMAP1', 'UMAP2'])
umap_df['Building_Type'] = df['Building_Type']

plt.figure(figsize=(10, 8))
sns.scatterplot(x='UMAP1', y='UMAP2', hue='Building_Type', data=umap_df, palette='viridis')
plt.title('UMAP Projection of the Electricity Bill Dataset')
plt.savefig(os.path.join(umap_dir, "umap_projection.png"))
plt.show()
