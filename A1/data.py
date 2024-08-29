import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('Electricity BILL.csv')

# Split the dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Training set size: {train_df.shape}")
print(f"Testing set size: {test_df.shape}")
