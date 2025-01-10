'''
The following code is part of the journey to learn data science. The code is written in Python and uses the following libraries: pandas, numpy, matplotlib, seaborn, and scikit-learn. 
The code is written to understand the data processing steps and to build a predictive model for wine quality prediction. While the experience was fruitful, the code is a mess and needs to be refactored.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR




# Loading datasets
w_wine_data = pd.read_csv('DATA/winequality-white.csv', sep=';')
r_wine_data = pd.read_csv('DATA/winequality-red.csv', sep=';')


# Checking for missing values
#print(w_wine_data.isnull().sum())
#print(r_wine_data.isnull().sum())

#Standardize Data

scaler = StandardScaler()
red_wine_scaled = scaler.fit_transform(r_wine_data.iloc[:, :-1])
white_wine_scaled = scaler.fit_transform(w_wine_data.iloc[:, :-1])

# Through visualization, we can see that the data is not normally distributed. Almost all the features are right-skewed. Hence we will try to fix this by following the below steps:

skewed_features = [col for col in r_wine_data.columns[:-1] if r_wine_data[col].skew() > 0.5]

# Apply log transformation to reduce skewness
for col in skewed_features:
    r_wine_data[col + '_log'] = np.log1p(r_wine_data[col])

# Visualize transformed distributions
for col in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(r_wine_data[col + '_log'], kde=True, bins=30)
    plt.title(f"Transformed Distribution of {col} (Log Scale)")
    plt.xlabel(f"{col} (Log Transformed)")
    plt.ylabel("Frequency")
    plt.show()
    
skewed_features = [col for col in w_wine_data.columns[:-1] if w_wine_data[col].skew() > 0.5]

# Apply log transformation to reduce skewness
for col in skewed_features:
    w_wine_data[col + '_log'] = np.log1p(w_wine_data[col])

# Visualize transformed distributions
for col in skewed_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(w_wine_data[col + '_log'], kde=True, bins=30)
    plt.title(f"Transformed Distribution of {col} (Log Scale)")
    plt.xlabel(f"{col} (Log Transformed)")
    plt.ylabel("Frequency")
    plt.show()

# Outlier detection and removal using IQR
for col in r_wine_data.columns[:-1]:  # Exclude 'quality'
    Q1 = r_wine_data[col].quantile(0.25)
    Q3 = r_wine_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
# Remove outliers
r_wine_data = r_wine_data[(r_wine_data[col] >= lower_bound) & (r_wine_data[col] <= upper_bound)]

# Visualize updated boxplots
for col in w_wine_data.columns[:-1]:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=w_wine_data[col])
    plt.title(f"Boxplot of {col} (After Outlier Removal)")
    plt.xlabel(col)
    plt.show()


# Outlier detection and removal using IQR
for col in w_wine_data.columns[:-1]:  # Exclude 'quality'
    Q1 = w_wine_data[col].quantile(0.25)
    Q3 = w_wine_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
# Remove outliers
w_wine_data = w_wine_data[(w_wine_data[col] >= lower_bound) & (w_wine_data[col] <= upper_bound)]

# Visualize updated boxplots
for col in w_wine_data.columns[:-1]:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=w_wine_data[col])
    plt.title(f"Boxplot of {col} (After Outlier Removal)")
    plt.xlabel(col)
    plt.show()


# Plot distribution of 'quality'
plt.figure(figsize=(8, 4))
sns.countplot(x=r_wine_data['quality'])
plt.title("Distribution of Wine Quality")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.show()

# Correlation heatmap with 'quality'
plt.figure(figsize=(12, 8))
corr_with_quality = r_wine_data.corr()['quality'].sort_values(ascending=False)
sns.heatmap(corr_with_quality.to_frame(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation of Features with Wine Quality")
plt.show()

# Grouped statistics by quality
grouped_stats = r_wine_data.groupby('quality').mean()
print(grouped_stats)

# Visualize feature trends across quality levels
for col in r_wine_data.columns[:-1]:
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=grouped_stats, x=grouped_stats.index, y=col)
    plt.title(f"Trend of {col} Across Wine Quality Levels")
    plt.xlabel("Quality")
    plt.ylabel(col)
    plt.show()


# Plot distribution of 'quality'
plt.figure(figsize=(8, 4))
sns.countplot(x=w_wine_data['quality'])
plt.title("Distribution of Wine Quality")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.show()

# Correlation heatmap with 'quality'
plt.figure(figsize=(12, 8))
corr_with_quality = w_wine_data.corr()['quality'].sort_values(ascending=False)
sns.heatmap(corr_with_quality.to_frame(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation of Features with Wine Quality")
plt.show()

# Grouped statistics by quality
grouped_stats = w_wine_data.groupby('quality').mean()
print(grouped_stats)

# Visualize feature trends across quality levels
for col in w_wine_data.columns[:-1]:
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=grouped_stats, x=grouped_stats.index, y=col)
    plt.title(f"Trend of {col} Across Wine Quality Levels")
    plt.xlabel("Quality")
    plt.ylabel(col)
    plt.show()


#Pipeline 

# Select features (log-transformed features where applicable)
features = [
    'volatile acidity', 'citric acid', 'alcohol_log', 'sulphates_log',
    'chlorides_log', 'density_log'
]  # Adjust this based on correlation analysis
X = r_wine_data[features]
y = r_wine_data['quality']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
    "Support Vector Regression": SVR(kernel='rbf'),
}

# Train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - Mean Squared Error: {mse:.3f}, R2 Score: {r2:.3f}")

# Ensure all features exist
required_features = [
    'volatile acidity', 'citric acid', 'alcohol_log', 
    'sulphates_log', 'chlorides_log', 'density'
]

# Check if all features exist in the dataset
missing_features = [f for f in required_features if f not in r_wine_data.columns]
if missing_features:
    print(f"Missing features: {missing_features}. Ensure preprocessing is correct.")
else:
    # Select features and target
    X = r_wine_data[required_features]
    y = r_wine_data['quality']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models for comparison
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
        "Support Vector Regression": SVR(kernel='rbf'),
    }

    # Train and evaluate models
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} - Mean Squared Error: {mse:.3f}, R2 Score: {r2:.3f}")

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the model
rf = RandomForestRegressor(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', verbose=2, n_jobs=-1)

# Fit the model
print("Starting Grid Search...")
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
# Best hyperparameters
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate on the test set
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - Mean Squared Error: {mse_rf:.3f}, R2 Score: {r2_rf:.3f}")

# Evaluating the feature importances
# Feature Importance
importances = best_model.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Print feature importance
print(importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title("Feature Importance in Random Forest Model")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# Predicted vs. Actual Values
y_test_pred = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)  # Ideal fit line
plt.title("Predicted vs Actual Values")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.show()

# Save the model

import joblib

# Save the model
joblib.dump(best_model, "random_forest_tuned_model.pkl")
print("Model saved as random_forest_tuned_model.pkl")

# To load the model later
# loaded_model = joblib.load("random_forest_tuned_model.pkl")

