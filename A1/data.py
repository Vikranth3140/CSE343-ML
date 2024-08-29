import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('Electricity Bill.csv')

# Split the dataset into 80:20 train and test splits
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Training set size: {train_df.shape}")
print(f"Testing set size: {test_df.shape}")

# EDA

# Pair Plots
def create_pair_plots(df, feature_group, title):
    sns.pairplot(df[feature_group], hue='Building_Type')
    plt.suptitle(title, y=1.02)
    plt.show()

# Building characteristics features
group_1 = ['Construction_Year', 'Number_of_Floors', 'Energy_Consumption_Per_SqM', 'Number_of_Residents']
create_pair_plots(df, group_1, "Pair Plot - Building Characteristics")

# Environmental factors features
group_2 = ['Water_Usage_Per_Building', 'Waste_Recycled_Percentage', 'Indoor_Air_Quality', 'Energy_Per_SqM']
create_pair_plots(df, group_2, "Pair Plot - Environmental Factors")

# Energy and maintenance features
group_3 = ['Electricity_Bill', 'Energy_Consumption_Per_SqM', 'Maintenance_Resolution_Time', 'Smart_Devices_Count']
create_pair_plots(df, group_3, "Pair Plot - Energy and Maintenance")

# Box plots for each numerical feature
plt.figure(figsize=(12, 8))
df.boxplot()
plt.xticks(rotation=45)
plt.title("Box Plot for Numerical Features")
plt.show()

# Violin Plots
def create_violin_plots(df, features, category):
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=category, y=feature, data=df)
        plt.title(f"Violin Plot - {feature} by {category}")
        plt.show()

# Violin Plots

# Building characteristics features
create_violin_plots(df, group_1, 'Building_Type')

# Environmental factors features
create_violin_plots(df, group_2, 'Building_Type')

# Energy and maintenance features
group_4 = ['Electricity_Bill', 'Maintenance_Resolution_Time', 'Smart_Devices_Count']
create_violin_plots(df, group_4, 'Building_Type')

# Count plot for categorical features
def create_count_plots(df, features):
    plt.figure(figsize=(20, 15))
    
    for i, feature in enumerate(features, 1):
        plt.subplot(2, 2, i)
        sns.countplot(x=feature, data=df)
        plt.title(f"Count Plot - {feature}")
    
    plt.tight_layout()
    plt.show()

# Count plots
categorical_features = ['Building_Type', 'Green_Certified', 'Building_Status', 'Maintenance_Priority']
create_count_plots(df, categorical_features)

# Correlation heatmap
numeric_df = df.select_dtypes(include='number')
corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
