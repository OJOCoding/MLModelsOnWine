import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

# Load the dataset
w_wine_data = pd.read_csv('Datasets/winequality-white.csv', sep=';')

# Feature Engineering
features = [
    'volatile acidity', 'citric acid', 'alcohol', 'sulphates', 'chlorides', 'density'
]
X = w_wine_data[features]
y = w_wine_data['quality']

# Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest with RandomizedSearchCV
rf = RandomForestRegressor(random_state=42)

# Hyperparameter grid
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Randomized search
rf_search = RandomizedSearchCV(
    rf, param_distributions, n_iter=50, scoring='neg_mean_squared_error',
    cv=5, random_state=42, n_jobs=-1
)

print("Starting Randomized Search for Random Forest...")
rf_search.fit(X_train_scaled, y_train)

# Best model
best_rf = rf_search.best_estimator_
best_params = rf_search.best_params_

# Evaluate the best model
y_pred = best_rf.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Best Parameters for Random Forest: {best_params}")
print("Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R2 Score: {r2:.3f}")

# Save the best model
model_save_path = 'SavedModels/RandomForest_WhiteWine_Best.pkl'
dump(best_rf, model_save_path)
print(f"Best Random Forest model saved at '{model_save_path}'")
