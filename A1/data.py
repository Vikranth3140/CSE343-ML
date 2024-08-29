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

# Pair Plot

# Building characteristics features
group_1 = df[['Construction_Year', 'Number_of_Floors', 'Energy_Consumption_Per_SqM', 'Number_of_Residents']]
sns.pairplot(group_1)
plt.suptitle("Pair Plot - Building Characteristics", y=1.02)
plt.show()

# Environmental factors features
group_2 = df[['Water_Usage_Per_Building', 'Waste_Recycled_Percentage', 'Indoor_Air_Quality', 'Energy_Per_SqM']]
sns.pairplot(group_2)
plt.suptitle("Pair Plot - Environmental Factors", y=1.02)
plt.show()

# Energy and maintenance features
group_3 = df[['Electricity_Bill', 'Energy_Consumption_Per_SqM', 'Maintenance_Resolution_Time', 'Smart_Devices_Count']]
sns.pairplot(group_3)
plt.suptitle("Pair Plot - Energy and Maintenance", y=1.02)
plt.show()


# Box plots for each numerical feature
plt.figure(figsize=(12, 8))
df.boxplot()
plt.xticks(rotation=45)
plt.show()

# Violin plot for a particular categorical feature

# Building characteristics features
plt.figure(figsize=(10, 6))
sns.violinplot(x='Building_Type', y='Construction_Year', data=df)
plt.title("Violin Plot - Construction Year by Building Type")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='Building_Type', y='Number_of_Floors', data=df)
plt.title("Violin Plot - Number of Floors by Building Type")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='Building_Type', y='Energy_Consumption_Per_SqM', data=df)
plt.title("Violin Plot - Energy Consumption Per SqM by Building Type")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='Building_Type', y='Number_of_Residents', data=df)
plt.title("Violin Plot - Number of Residents by Building Type")
plt.show()

# Environmental factors features
plt.figure(figsize=(10, 6))
sns.violinplot(x='Building_Type', y='Water_Usage_Per_Building', data=df)
plt.title("Violin Plot - Water Usage Per Building by Building Type")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='Building_Type', y='Waste_Recycled_Percentage', data=df)
plt.title("Violin Plot - Waste Recycled Percentage by Building Type")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='Building_Type', y='Indoor_Air_Quality', data=df)
plt.title("Violin Plot - Indoor Air Quality by Building Type")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='Building_Type', y='Energy_Per_SqM', data=df)
plt.title("Violin Plot - Energy Per SqM by Building Type")
plt.show()


# Energy and maintenance features
plt.figure(figsize=(10, 6))
sns.violinplot(x='Building_Type', y='Electricity_Bill', data=df)
plt.title("Violin Plot - Electricity Bill by Building Type")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='Building_Type', y='Maintenance_Resolution_Time', data=df)
plt.title("Violin Plot - Maintenance Resolution Time by Building Type")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='Building_Type', y='Smart_Devices_Count', data=df)
plt.title("Violin Plot - Smart Devices Count by Building Type")
plt.show()


# Count plot for categorical features

plt.figure(figsize=(20, 15))

# Building_Type
plt.subplot(2, 2, 1)
sns.countplot(x='Building_Type', data=df)
plt.title("Count Plot - Building Type")

# Green_Certified
plt.subplot(2, 2, 2)
sns.countplot(x='Green_Certified', data=df)
plt.title("Count Plot - Green Certified")

# Building_Status
plt.subplot(2, 2, 3)
sns.countplot(x='Building_Status', data=df)
plt.title("Count Plot - Building Status")

# Maintenance_Priority
plt.subplot(2, 2, 4)
sns.countplot(x='Maintenance_Priority', data=df)
plt.title("Count Plot - Maintenance Priority")

plt.tight_layout()
plt.show()


# Correlation heatmap
numeric_df = df.select_dtypes(include='number')
corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
