import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer

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

# Split the dataset into 80:20 train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

def adjusted_r2(r2, n, k):
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))

# Evaluation Metrics
def evaluate_model(y_true, y_pred, X):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = adjusted_r2(r2, X.shape[0], X.shape[1])
    
    return mse, rmse, mae, r2, adj_r2

# Apply Gradient Boosting Regressor
gbr_model = GradientBoostingRegressor(random_state=42)
gbr_model.fit(X_train_preprocessed, y_train)

y_train_pred_gbr = gbr_model.predict(X_train_preprocessed)
y_test_pred_gbr = gbr_model.predict(X_test_preprocessed)

train_mse_gbr, train_rmse_gbr, train_mae_gbr, train_r2_gbr, train_adj_r2_gbr = evaluate_model(y_train, y_train_pred_gbr, X_train_preprocessed)
test_mse_gbr, test_rmse_gbr, test_mae_gbr, test_r2_gbr, test_adj_r2_gbr = evaluate_model(y_test, y_test_pred_gbr, X_test_preprocessed)

print("Train Metrics (Gradient Boosting Regressor):")
print(f"  MSE: {train_mse_gbr:.4f}")
print(f"  RMSE: {train_rmse_gbr:.4f}")
print(f"  MAE: {train_mae_gbr:.4f}")
print(f"  R²: {train_r2_gbr:.4f}")
print(f"  Adjusted R²: {train_adj_r2_gbr:.4f}")

print("\nTest Metrics (Gradient Boosting Regressor):")
print(f"  MSE: {test_mse_gbr:.4f}")
print(f"  RMSE: {test_rmse_gbr:.4f}")
print(f"  MAE: {test_mae_gbr:.4f}")
print(f"  R²: {test_r2_gbr:.4f}")
print(f"  Adjusted R²: {test_adj_r2_gbr:.4f}")