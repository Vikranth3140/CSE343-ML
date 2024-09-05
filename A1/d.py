import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import RFE

df = pd.read_csv('Electricity Bill.csv')
df.columns = df.columns.str.strip()

df.fillna(df.median(numeric_only=True), inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Normalize numerical features
numerical_features = df.select_dtypes(include=np.number).columns
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Encode categorical features using LabelEncoder
label_encoders = {}
categorical_features = df.select_dtypes(include='object').columns
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare the data for feature selection and Linear Regression
X = df.drop(columns=['Electricity_Bill'])  # Drop the target variable
y = df['Electricity_Bill']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Recursive Feature Elimination (RFE) to select 3 most important features
model = LinearRegression()
rfe = RFE(estimator=model, n_features_to_select=3)
rfe.fit(X_train, y_train)

# Get the selected features
selected_features = X_train.columns[rfe.support_]
print(f"Selected features by RFE: {selected_features}")

# Train the Linear Regression model using only the selected features
X_train_rfe = X_train[selected_features]
X_test_rfe = X_test[selected_features]

model.fit(X_train_rfe, y_train)

# Predictions
y_train_pred_rfe = model.predict(X_train_rfe)
y_test_pred_rfe = model.predict(X_test_rfe)

# Evaluation Metrics
def adjusted_r2(r2, n, k):
    """Function to calculate the adjusted R² score."""
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))

def evaluate_model(y_true, y_pred, X):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = adjusted_r2(r2, X.shape[0], X.shape[1])
    
    return mse, rmse, mae, r2, adj_r2

# Train Metrics with RFE selected features
train_mse_rfe, train_rmse_rfe, train_mae_rfe, train_r2_rfe, train_adj_r2_rfe = evaluate_model(y_train, y_train_pred_rfe, X_train_rfe)

# Test Metrics with RFE selected features
test_mse_rfe, test_rmse_rfe, test_mae_rfe, test_r2_rfe, test_adj_r2_rfe = evaluate_model(y_test, y_test_pred_rfe, X_test_rfe)

# Print the results for the model using RFE selected features
print("\nTrain Metrics (with RFE selected features):")
print(f"  MSE: {train_mse_rfe:.4f}")
print(f"  RMSE: {train_rmse_rfe:.4f}")
print(f"  MAE: {train_mae_rfe:.4f}")
print(f"  R²: {train_r2_rfe:.4f}")
print(f"  Adjusted R²: {train_adj_r2_rfe:.4f}")

print("\nTest Metrics (with RFE selected features):")
print(f"  MSE: {test_mse_rfe:.4f}")
print(f"  RMSE: {test_rmse_rfe:.4f}")
print(f"  MAE: {test_mae_rfe:.4f}")
print(f"  R²: {test_r2_rfe:.4f}")
print(f"  Adjusted R²: {test_adj_r2_rfe:.4f}")
