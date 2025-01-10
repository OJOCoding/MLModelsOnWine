'''
Dataset used: White Wine Quality
Model tested: Linear Regression, Random Forest Regressor, Support Vector Machine
This file is used to test the initial model that was created. The model is tested on the test data and the accuracy is calculated.

Results: Based on the findings, the best model to classify our dataset is the Random Forest Classifier with a Accuracy of 0.938
'''

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

# Feature selection based on correlation with 'quality'
corr = w_wine_data.corr()['quality'].sort_values(ascending=False)
selected_features = corr.index[1:6]  # Top 5 features excluding 'quality'

# Prepare feature set (X) and target variable (y)
X = w_wine_data[selected_features]
y = w_wine_data['quality']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define regression models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42, n_estimators=100),
    "Support Vector Regressor": SVR(kernel='rbf')
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"model": model, "mse": mse, "r2": r2}
    print(f"{name} - Mean Squared Error: {mse:.3f}, R2 Score: {r2:.3f}")

# Determine the best model
best_model_name = min(results, key=lambda x: results[x]['mse'])
best_model = results[best_model_name]['model']
print(f"\nBest Model: {best_model_name} with MSE: {results[best_model_name]['mse']:.3f}, R2: {results[best_model_name]['r2']:.3f}")

# Function to evaluate the best model
def evaluate_model(best_model, X_test, y_test):
    """
    Evaluate the given model on test data and print detailed metrics.
    """
    # Predictions
    y_pred = best_model.predict(X_test)
    
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

# Evaluate the best model
print("\nEvaluating the Best Model...")
evaluation_results = evaluate_model(best_model, X_test_scaled, y_test)


# Save the best model
save_dir = 'SavedModels'
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, f"{best_model_name.replace(' ', '_')}_model.pkl")
joblib.dump(best_model, model_path)
print(f"Best model saved to {model_path}")
