import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('Electricity Bill.csv')
df.columns = df.columns.str.strip()

# Handling missing values

# Filling numerical values with the median
df.fillna(df.median(numeric_only=True), inplace=True)

# Filling categorical missing values with mode
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

X = df.drop(columns=['Electricity_Bill'])
y = df['Electricity_Bill']

categorical_features = X.select_dtypes(include='object').columns.tolist()
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

# One-Hot Encoding for categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)])

# Apply Ridge Regression in a pipeline
ridge_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('ridge', Ridge(alpha=1.0))
])

# Split the dataset into 80:20 train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Ridge Regression model
ridge_model.fit(X_train, y_train)

y_train_pred_ridge = ridge_model.predict(X_train)
y_test_pred_ridge = ridge_model.predict(X_test)

# Evaluation Metrics
def adjusted_r2(r2, n, k):
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))

def evaluate_model(y_true, y_pred, X):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = adjusted_r2(r2, X.shape[0], X.shape[1])
    
    return mse, rmse, mae, r2, adj_r2

train_mse_ridge, train_rmse_ridge, train_mae_ridge, train_r2_ridge, train_adj_r2_ridge = evaluate_model(y_train, y_train_pred_ridge, X_train)
test_mse_ridge, test_rmse_ridge, test_mae_ridge, test_r2_ridge, test_adj_r2_ridge = evaluate_model(y_test, y_test_pred_ridge, X_test)

print("Train Metrics (Ridge Regression with One-Hot Encoding):")
print(f"  MSE: {train_mse_ridge:.4f}")
print(f"  RMSE: {train_rmse_ridge:.4f}")
print(f"  MAE: {train_mae_ridge:.4f}")
print(f"  R²: {train_r2_ridge:.4f}")
print(f"  Adjusted R²: {train_adj_r2_ridge:.4f}")

print("\nTest Metrics (Ridge Regression with One-Hot Encoding):")
print(f"  MSE: {test_mse_ridge:.4f}")
print(f"  RMSE: {test_rmse_ridge:.4f}")
print(f"  MAE: {test_mae_ridge:.4f}")
print(f"  R²: {test_r2_ridge:.4f}")
print(f"  Adjusted R²: {test_adj_r2_ridge:.4f}")