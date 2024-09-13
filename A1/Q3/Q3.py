import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

eda_dir = 'EDA/'
os.makedirs(eda_dir, exist_ok=True)

df = pd.read_csv('Electricity Bill.csv')
df.columns = df.columns.str.strip()

# Split the dataset into 80:20 train and test splits
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Training set size: {train_df.shape}")
print(f"Testing set size: {test_df.shape}")

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



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('Electricity Bill.csv')
df.columns = df.columns.str.strip()

# Handling missing values

null_values = df.isnull().sum()
print("\nNumber of Null Values in Each Column:\n")
print(f"{null_values}\n")

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

# Split the dataset into 80:20 train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

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

train_mse, train_rmse, train_mae, train_r2, train_adj_r2 = evaluate_model(y_train, y_train_pred, X_train)
test_mse, test_rmse, test_mae, test_r2, test_adj_r2 = evaluate_model(y_test, y_test_pred, X_test)

print("Train Metrics:")
print(f"  MSE: {train_mse}")
print(f"  RMSE: {train_rmse}")
print(f"  MAE: {train_mae}")
print(f"  R²: {train_r2}")
print(f"  Adjusted R²: {train_adj_r2}")

print("\nTest Metrics:")
print(f"  MSE: {test_mse}")
print(f"  RMSE: {test_rmse}")
print(f"  MAE: {test_mae}")
print(f"  R²: {test_r2}")
print(f"  Adjusted R²: {test_adj_r2}")



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




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
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
print(f"  MSE: {train_mse_ridge}")
print(f"  RMSE: {train_rmse_ridge}")
print(f"  MAE: {train_mae_ridge}")
print(f"  R²: {train_r2_ridge}")
print(f"  Adjusted R²: {train_adj_r2_ridge}")

print("\nTest Metrics (Ridge Regression with One-Hot Encoding):")
print(f"  MSE: {test_mse_ridge}")
print(f"  RMSE: {test_rmse_ridge}")
print(f"  MAE: {test_mae_ridge}")
print(f"  R²: {test_r2_ridge}")
print(f"  Adjusted R²: {test_adj_r2_ridge}")




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
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
    print(f"Train MSE: {res['train_mse']}, RMSE: {res['train_rmse']}, MAE: {res['train_mae']}, R²: {res['train_r2']}, Adjusted R²: {res['train_adj_r2']}")
    print(f"Test  MSE: {res['test_mse']}, RMSE: {res['test_rmse']}, MAE: {res['test_mae']}, R²: {res['test_r2']}, Adjusted R²: {res['test_adj_r2']}")



    import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import ElasticNet
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

def perform_elastic_net(alpha, X_train, X_test, y_train, y_test):
    # Apply ElasticNet
    elastic_net = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42)
    elastic_net.fit(X_train, y_train)
    
    y_train_pred = elastic_net.predict(X_train)
    y_test_pred = elastic_net.predict(X_test)
    
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
    
    train_mse, train_rmse, train_mae, train_r2, train_adj_r2 = evaluate_model(y_train, y_train_pred, X_train)
    test_mse, test_rmse, test_mae, test_r2, test_adj_r2 = evaluate_model(y_test, y_test_pred, X_test)
    
    return {
        "alpha": alpha,
        "train_mse": train_mse, "train_rmse": train_rmse, "train_mae": train_mae, "train_r2": train_r2, "train_adj_r2": train_adj_r2,
        "test_mse": test_mse, "test_rmse": test_rmse, "test_mae": test_mae, "test_r2": test_r2, "test_adj_r2": test_adj_r2
    }

# Try ElasticNet with different alpha values
alpha_values = [0.1, 0.05, 0.01, 0.005, 0.001]
results = []

for alpha in alpha_values:
    result = perform_elastic_net(alpha, X_train_preprocessed, X_test_preprocessed, y_train, y_test)
    results.append(result)

for res in results:
    print(f"\nResults for alpha={res['alpha']}:")
    print(f"Train MSE: {res['train_mse']}, RMSE: {res['train_rmse']}, MAE: {res['train_mae']}, R²: {res['train_r2']}, Adjusted R²: {res['train_adj_r2']}")
    print(f"Test  MSE: {res['test_mse']}, RMSE: {res['test_rmse']}, MAE: {res['test_mae']}, R²: {res['test_r2']}, Adjusted R²: {res['test_adj_r2']}")




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
print(f"  MSE: {train_mse_gbr}")
print(f"  RMSE: {train_rmse_gbr}")
print(f"  MAE: {train_mae_gbr}")
print(f"  R²: {train_r2_gbr}")
print(f"  Adjusted R²: {train_adj_r2_gbr}")

print("\nTest Metrics (Gradient Boosting Regressor):")
print(f"  MSE: {test_mse_gbr}")
print(f"  RMSE: {test_rmse_gbr}")
print(f"  MAE: {test_mae_gbr}")
print(f"  R²: {test_r2_gbr}")
print(f"  Adjusted R²: {test_adj_r2_gbr}")