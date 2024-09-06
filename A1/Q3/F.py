import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
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

def perform_ica_and_regression(n_components, X_train, X_test, y_train, y_test):
    # Apply ICA
    ica = FastICA(n_components=n_components, random_state=42)
    X_train_ica = ica.fit_transform(X_train)
    X_test_ica = ica.transform(X_test)
    
    # Apply Linear Regression
    model = LinearRegression()
    model.fit(X_train_ica, y_train)
    
    y_train_pred = model.predict(X_train_ica)
    y_test_pred = model.predict(X_test_ica)
    
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
    
    train_mse, train_rmse, train_mae, train_r2, train_adj_r2 = evaluate_model(y_train, y_train_pred, X_train_ica)
    test_mse, test_rmse, test_mae, test_r2, test_adj_r2 = evaluate_model(y_test, y_test_pred, X_test_ica)
    
    return {
        "n_components": n_components,
        "train_mse": train_mse, "train_rmse": train_rmse, "train_mae": train_mae, "train_r2": train_r2, "train_adj_r2": train_adj_r2,
        "test_mse": test_mse, "test_rmse": test_rmse, "test_mae": test_mae, "test_r2": test_r2, "test_adj_r2": test_adj_r2
    }

# ICA with 4, 5, 6, and 8 components
ica_components = [4, 5, 6, 8]
results = []

for n in ica_components:
    result = perform_ica_and_regression(n, X_train_preprocessed, X_test_preprocessed, y_train, y_test)
    results.append(result)

for res in results:
    print(f"\nResults for {res['n_components']} components:")
    print(f"Train MSE: {res['train_mse']:.4f}, RMSE: {res['train_rmse']:.4f}, MAE: {res['train_mae']:.4f}, R²: {res['train_r2']:.4f}, Adjusted R²: {res['train_adj_r2']:.4f}")
    print(f"Test  MSE: {res['test_mse']:.4f}, RMSE: {res['test_rmse']:.4f}, MAE: {res['test_mae']:.4f}, R²: {res['test_r2']:.4f}, Adjusted R²: {res['test_adj_r2']:.4f}")