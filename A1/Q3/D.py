import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('Electricity Bill.csv')
df.columns = df.columns.str.strip()

# Handling missing values

# Filling numerical values with the median
df.fillna(df.median(numeric_only=True), inplace=True)

# Filling categorical missing values with mode
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# # Normalizing numerical features
# numerical_features = df.select_dtypes(include=np.number).columns
# scaler = StandardScaler()
# df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Normalizing target variable (y)
scaler = StandardScaler()
y = scaler.fit_transform(df[['Electricity_Bill']])

# Encoding categorical features using LabelEncoder
label_encoders = {}
categorical_features = df.select_dtypes(include='object').columns
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=['Electricity_Bill'])
y = df['Electricity_Bill']

# Apply Linear Regression with Recursive Feature Elimination (RFE) to select the 3 most important features
model = LinearRegression()
rfe = RFE(model, n_features_to_select=3)
rfe.fit(X, y)

selected_features = X.columns[rfe.support_]
print("Selected Features:", selected_features)

X_selected_train, X_selected_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)

model.fit(X_selected_train, y_train)

y_train_pred = model.predict(X_selected_train)
y_test_pred = model.predict(X_selected_test)

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

train_mse, train_rmse, train_mae, train_r2, train_adj_r2 = evaluate_model(y_train, y_train_pred, X_selected_train)
test_mse, test_rmse, test_mae, test_r2, test_adj_r2 = evaluate_model(y_test, y_test_pred, X_selected_test)

print("Train Metrics (with selected features):")
print(f"  MSE: {train_mse}")
print(f"  RMSE: {train_rmse}")
print(f"  MAE: {train_mae}")
print(f"  R²: {train_r2}")
print(f"  Adjusted R²: {train_adj_r2}")

print("\nTest Metrics (with selected features):")
print(f"  MSE: {test_mse}")
print(f"  RMSE: {test_rmse}")
print(f"  MAE: {test_mae}")
print(f"  R²: {test_r2}")
print(f"  Adjusted R²: {test_adj_r2}")