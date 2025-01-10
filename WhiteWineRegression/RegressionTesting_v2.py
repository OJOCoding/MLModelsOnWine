# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import joblib

# Load the white wine dataset
w_wine_data = pd.read_csv('Datasets/winequality-white.csv', sep=';')
# Use the same features as in previous models
features = ['volatile acidity', 'citric acid', 'alcohol', 'sulphates', 'chlorides', 'density']
X = w_wine_data[features]
y = w_wine_data['quality']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest parameter tuning
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
rf = RandomForestRegressor(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='r2', n_jobs=-1)

# Train and evaluate Random Forest
print("Starting Grid Search for Random Forest...")
grid_search_rf.fit(X_train_scaled, y_train)
best_rf = grid_search_rf.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the given model on test data and print detailed metrics.
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print("\nModel Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"R2 Score: {r2:.3f}")
    
    # Visualize Actual vs Predicted
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    plt.title("Actual vs Predicted Values")
    plt.xlabel("Actual Quality")
    plt.ylabel("Predicted Quality")
    plt.grid()
    plt.show()
    
    return {"mse": mse, "r2": r2, "rmse": rmse}

# Evaluate the tuned model
print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
evaluation_results_rf = evaluate_model(best_rf, X_test_scaled, y_test)

# Support Vector Regression parameter tuning
param_grid_svr = {
    'kernel': ['rbf', 'linear'],
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.2, 0.3],
}
svr = SVR()
grid_search_svr = GridSearchCV(estimator=svr, param_grid=param_grid_svr, cv=5, scoring='r2', n_jobs=-1)

# Train and evaluate SVR
print("Starting Grid Search for SVR...")
grid_search_svr.fit(X_train_scaled, y_train)
best_svr = grid_search_svr.best_estimator_

# Evaluate the tuned model
print("Best Parameters for SVR:", grid_search_svr.best_params_)
evaluation_results_svr = evaluate_model(best_svr, X_test_scaled, y_test)
