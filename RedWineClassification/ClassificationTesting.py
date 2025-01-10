'''
Dataset used: Red Wine Quality
Model tested: Logistic Regression,Random Forest Classifier, Support Vector Machine
This file is used to test the initial model that was created. The model is tested on the test data and the accuracy is calculated.

Results: Based on the findings, the best model to classify our dataset is the Random Forest Classifier with a Accuracy of 0.938. 

Conclussions: The classification model achieved an accuracy of 93.8%, which is quite high. This suggests that the dataset is well-suited for classification tasks. The Random Forest Classifier performed the best among the models tested, indicating that it is a good choice for this dataset.
'''

# Importing necessary libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Load the red wine dataset
r_wine_data = pd.read_csv('Datasets/winequality-red.csv', sep=';')

# Convert quality into binary target
r_wine_data['quality_label'] = (r_wine_data['quality'] >= 7).astype(int)  # 1 for high quality, 0 otherwise
r_wine_data = r_wine_data.drop(columns=['quality'])  # Drop the original quality column

# Data processing and preparation

# Define features and target
X = r_wine_data.drop(columns=['quality_label'])
#other way to define X is X = r_wine_data.iloc[:, :-1]
y = r_wine_data['quality_label']


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training and evaluation

def find_best_model(models, X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Trains and evaluates multiple models and returns the best model based on accuracy (and optionally ROC-AUC).

    Parameters:
    - models (dict): Dictionary of models to evaluate
    - X_train_scaled (array): Scaled training features
    - X_test_scaled (array): Scaled test features
    - y_train (array): Training target labels
    - y_test (array): Test target labels

    Returns:
    - best_model (model): The best performing model
    - results (dict): Dictionary of model names and their performance metrics
    """
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        
        print(f"{name} - Accuracy: {accuracy:.3f}, ROC-AUC: {roc_auc:.3f}" if roc_auc else f"{name} - Accuracy: {accuracy:.3f}")
        
        # Save results
        results[name] = {'accuracy': accuracy, 'roc_auc': roc_auc}

    # Select the best model based on Accuracy (and ROC-AUC as a secondary metric)
    best_model_name = max(results, key=lambda x: results[x]['accuracy'] if results[x]['accuracy'] else -1)
    best_model = models[best_model_name]
    
    print(f"\nBest Model: {best_model_name} with Accuracy: {results[best_model_name]['accuracy']:.3f}")
    
    return best_model, results

# Define models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Support Vector Machine": SVC(kernel='rbf', probability=True, random_state=42)
}

# Find the best model based on accuracy
best_model, results = find_best_model(models, X_train_scaled, X_test_scaled, y_train, y_test)

# Results visualization
y_pred_best = best_model.predict(X_test_scaled)

# Confusion Matrix for the Best Model
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Quality', 'High Quality'], yticklabels=['Low Quality', 'High Quality'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - Best Model ({best_model.__class__.__name__})')
plt.show()

# Save the best model

def save_best_model(best_model):
    # Ensure 'SavedModels' folder exists
    os.makedirs('SavedModels', exist_ok=True)
    
    # Define the path for saving the model
    model_filename = f'SavedModels/{best_model.__class__.__name__}_model.pkl'
    
    # Save the model using joblib
    joblib.dump(best_model, model_filename)
    
    # Print confirmation
    print(f"Model saved as {model_filename}")

# Save the best model after finding it
save_best_model(best_model)
